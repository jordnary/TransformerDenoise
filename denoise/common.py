"""Data, deterministic noise, metrics, and inference shared by the commands."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

SIGMA = 50 / 255


def resolve_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or --device auto")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def setup(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def dataset(root, train):
    ds = MNIST(root, train=train, download=True)
    return ds.data[:, None], ds.targets


def split_indices(size, validation_size, seed):
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    return order[validation_size:], order[:validation_size]


def add_noise(clean, generator=None):
    return clean + SIGMA * torch.randn(clean.shape, device=clean.device, generator=generator)


def tone_augment(clean, probability, generator):
    draws = torch.rand((4, len(clean), 1, 1, 1), device=clean.device, generator=generator)
    contrast = 0.25 + 0.75 * draws[0]
    base = torch.where(draws[1] < 0.5, 1 - clean, clean)
    transformed = contrast * base + (1 - contrast) * draws[2]
    return torch.where(draws[3] < probability, transformed, clean)


def structure_augment(clean, probability, generator):
    """Affine crops and convex mixtures using only the current clean batch."""
    if not 0 <= probability <= 1:
        raise ValueError("Augmentation probability must be in [0, 1]")
    if probability == 0:
        return clean
    n = len(clean)
    draws = torch.rand((2 * n, 4), device=clean.device, generator=generator)
    angle, scale = draws[:, 0] * (2 * torch.pi), 0.7 + 1.3 * draws[:, 1]
    theta = clean.new_zeros((2 * n, 2, 3))
    theta[:, 0, 0] = theta[:, 1, 1] = angle.cos() * scale
    theta[:, 0, 1], theta[:, 1, 0] = -angle.sin() * scale, angle.sin() * scale
    theta[:, :, 2] = (draws[:, 2:] * 2 - 1) * 0.65
    source = torch.cat((clean, clean.flip(0)))
    grid = F.affine_grid(theta, source.shape, align_corners=False)
    first, second = F.grid_sample(source, grid, align_corners=False).chunk(2)
    choices = torch.rand((3, n, 1, 1, 1), device=clean.device, generator=generator)
    weight = 0.25 + 0.5 * choices[0]
    mixed = weight * first + (1 - weight) * second
    transformed = torch.where(choices[1] < 0.5, mixed, first)
    return torch.where(choices[2] < probability, transformed, clean)


def structure_validation(images, seed):
    generator = torch.Generator().manual_seed(seed)
    clean = F.pad(images.float() / 255, (2,) * 4)
    return tone_augment(structure_augment(clean, 1, generator), 1, generator)


def fixed_noise(shape, seed):
    return SIGMA * torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def scores(pred, clean):
    error = (pred.double() - clean.double()).flatten(1)
    mse = error.square().mean(1)
    return {"l1": error.abs().mean(1), "mse": mse, "psnr": -10 * torch.log10(mse)}


def summarize(metrics):
    result = {key: float(value.mean()) for key, value in metrics.items()}
    psnr = metrics["psnr"]
    result["psnr_std"] = float(psnr.std(unbiased=len(psnr) > 1))
    result["psnr_standard_error"] = result["psnr_std"] / len(psnr) ** 0.5
    result["psnr_from_mse"] = float(-10 * torch.log10(metrics["mse"].mean()))
    return result


def first_indices(labels):
    return torch.tensor([int(torch.where(labels == digit)[0][0]) for digit in range(10)])


@torch.inference_mode()
def predict(model, noisy, ensemble=False):
    if not ensemble:
        return model(noisy).float()
    total = torch.zeros_like(noisy)
    for flip in (False, True):
        for k in range(4):
            z = noisy.flip(-1) if flip else noisy
            y = model(z.rot90(k, (-2, -1))).float().rot90(-k, (-2, -1))
            total += y.flip(-1) if flip else y
    return total / 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
