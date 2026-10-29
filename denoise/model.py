"""A residual Transformer with linear patch projections and shifted windows."""

import torch
from torch import nn
from torch.nn import functional as F


def partition(x, window):
    b, h, w, c = x.shape
    return x.reshape(b, h // window, window, w // window, window, c).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def reverse(x, window, h, w):
    return x.reshape(-1, h // window, w // window, window, window, x.shape[-1]).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, h, w, x.shape[-1])


def shift_mask(h, w, window, shift, device):
    labels = torch.zeros((1, h, w, 1), device=device)
    slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    for i, ys in enumerate(slices):
        for j, xs in enumerate(slices):
            labels[:, ys, xs] = i * 3 + j
    labels = partition(labels, window).squeeze(-1)
    return (labels.unsqueeze(1) != labels.unsqueeze(2)).float() * -1e4


class Attention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        delta = coords.flatten(1)[:, :, None] - coords.flatten(1)[:, None, :]
        delta = delta + window - 1
        self.register_buffer("index", delta[0] * (2 * window - 1) + delta[1])
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x, mask):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.bias[self.index].permute(2, 0, 1).to(q.dtype)
        if mask is not None:
            windows = mask.shape[0]
            bias = bias[None] + mask[:, None].to(q.dtype)
            bias = bias.repeat(b // windows, 1, 1, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(2 * dim, dim)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.ffn = Attention(dim, heads, window), SwiGLU(dim)

    def forward(self, x, mask):
        h, w = x.shape[1:3]
        z = self.norm1(x)
        if self.shift:
            z = torch.roll(z, (-self.shift, -self.shift), (1, 2))
        z = reverse(self.attn(partition(z, self.window), mask if self.shift else None), self.window, h, w)
        if self.shift:
            z = torch.roll(z, (self.shift, self.shift), (1, 2))
        x = x + z
        return x + self.ffn(self.norm2(x))


class Denoiser(nn.Module):
    def __init__(self, dim=64, depth=8, heads=4, patch=2, window=4, halo=0, output_halo=0):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads or patch < 1 or window < 2 or depth < 1 or min(halo, output_halo) < 0:
            raise ValueError("Invalid Transformer dimensions")
        self.patch, self.window, self.halo = patch, window, halo
        self.output_halo = output_halo
        self.embed = nn.Linear((patch + 2 * halo) ** 2, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, window, (i % 2) * (window // 2)) for i in range(depth))
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, patch * patch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if output_halo:
            self.refine = nn.Linear(2 * dim, (patch + 2 * output_halo) ** 2)
            nn.init.zeros_(self.refine.weight)
            nn.init.zeros_(self.refine.bias)

    def initialize(self, weights):
        """Allow only a zero-initialized refinement branch to be added."""
        missing = set(self.state_dict()) - set(weights)
        if missing not in (set(), {"refine.weight", "refine.bias"}):
            raise ValueError("Initialization architecture mismatch")
        self.load_state_dict({**self.state_dict(), **weights}, strict=True)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError("Expected grayscale BCHW images with both dimensions >= 8")
        b, _, h, w = x.shape
        p, unit = self.patch, self.patch * self.window
        pad_h, pad_w = (-h) % unit, (-w) % unit
        noisy = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        hh, ww = noisy.shape[-2] // p, noisy.shape[-1] // p
        kernel = p + 2 * self.halo
        z = F.pad(noisy, (self.halo,) * 4, mode="reflect") if self.halo else noisy
        z = z.unfold(2, kernel, p).unfold(3, kernel, p).permute(0, 2, 3, 1, 4, 5)
        z = z.reshape(b, hh, ww, kernel * kernel)
        z = shallow = self.embed(z)
        mask = shift_mask(hh, ww, self.window, self.window // 2, x.device)
        for block in self.blocks:
            z = block(z, mask)
        z = self.norm(z)
        residual = self.head(z).reshape(b, hh, ww, p, p).permute(0, 1, 3, 2, 4)
        residual = residual.reshape(b, 1, hh * p, ww * p)
        if self.output_halo:
            k = p + 2 * self.output_halo
            patches = self.refine(torch.cat((z, shallow), dim=-1)).flatten(1, 2).transpose(1, 2)
            options = dict(output_size=(hh * p, ww * p), kernel_size=k, padding=self.output_halo, stride=p)
            divisor = F.fold(torch.ones_like(patches[:1, :, :]), **options)
            residual = residual + F.fold(patches, **options) / divisor
        return (noisy + residual)[..., :h, :w]
