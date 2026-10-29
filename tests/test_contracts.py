"""Numerical and structural contracts for the denoising experiment."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from denoise.common import SIGMA, add_noise, first_indices, fixed_noise, predict, scores, split_indices
from denoise.model import Denoiser, partition, reverse, shift_mask
from denoise.evaluate import gaussian_baseline


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sizes_and_no_convolution(self):
        model = Denoiser(dim=16, depth=2, heads=2)
        conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        self.assertFalse(any(isinstance(m, conv_types) for m in model.modules()))
        for shape in ((28, 28), (32, 32), (28, 32), (29, 31)):
            x = torch.randn(2, 1, *shape)
            output = model(x)
            self.assertEqual(output.shape, x.shape)
            torch.testing.assert_close(output, x)
            overlap = Denoiser(dim=16, depth=2, heads=2, halo=2)
            torch.testing.assert_close(overlap(x), x)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 3, 28, 28))

    def test_nonzero_gradients_and_parameter_update(self):
        torch.manual_seed(4)
        model = Denoiser(dim=16, depth=2, heads=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        clean = torch.rand(2, 1, 28, 28)
        noisy = add_noise(clean)
        initial = F.l1_loss(model(noisy), clean).item()
        for _ in range(12):
            optimizer.zero_grad()
            loss = F.l1_loss(model(noisy), clean)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertLess(F.l1_loss(model(noisy), clean).item(), initial)
        self.assertGreater(float(model.blocks[0].attn.qkv.weight.grad.abs().sum()), 0)

    def test_window_inverse_and_wrap_mask(self):
        x = torch.randn(3, 16, 20, 7)
        torch.testing.assert_close(reverse(partition(x, 4), 4, 16, 20), x)
        mask = shift_mask(16, 16, 4, 2, "cpu")
        self.assertEqual(mask.shape, (16, 16, 16))
        self.assertTrue(torch.all(mask[0] == 0))
        self.assertTrue(torch.all(torch.diagonal(mask, dim1=1, dim2=2) == 0))
        self.assertLess(float(mask[-1, 0, -1]), -1000)

    def test_noise_distribution_and_unclipped_input(self):
        clean = torch.full((2000, 1, 28, 28), 0.5)
        noisy = add_noise(clean, torch.Generator().manual_seed(7))
        noise = noisy - clean
        self.assertLess(abs(float(noise.mean())), 0.001)
        self.assertAlmostEqual(float(noise.std()), SIGMA, delta=0.001)
        self.assertLess(float(noisy.min()), 0)
        self.assertGreater(float(noisy.max()), 1)
        torch.testing.assert_close(fixed_noise(clean.shape, 7), noise)

    def test_metrics_use_unit_peak_and_unclipped_values(self):
        clean = torch.zeros(2, 1, 28, 28)
        result = scores(torch.full_like(clean, -0.1), clean)
        torch.testing.assert_close(result["psnr"], torch.full((2,), 20.0, dtype=torch.float64))
        self.assertAlmostEqual(float(result["l1"].mean()), 0.1, places=6)
        self.assertTrue(torch.isinf(scores(clean, clean)["psnr"]).all())

    def test_disjoint_split_and_first_occurrence(self):
        train, validation = split_indices(60000, 5000, 3407)
        self.assertEqual(len(train), 55000)
        self.assertEqual(len(set(train.tolist()) & set(validation.tolist())), 0)
        self.assertEqual(len(set(train.tolist()) | set(validation.tolist())), 60000)
        labels = torch.tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2, 4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8])
        self.assertEqual(first_indices(labels).tolist(), [3, 2, 1, 18, 4, 8, 11, 0, 61, 7])

    def test_ensemble_inverse_and_batch_independence(self):
        torch.manual_seed(8)
        model = Denoiser(dim=16, depth=2, heads=2).eval()
        nn.init.normal_(model.head.weight, std=0.01)
        x = torch.randn(3, 1, 28, 32)
        torch.testing.assert_close(predict(nn.Identity(), x, True), x)
        full = predict(model, x, True)
        parts = torch.cat([predict(model, part, True) for part in x.split(1)])
        torch.testing.assert_close(full, parts, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(gaussian_baseline(torch.ones_like(x)), torch.ones_like(x))


if __name__ == "__main__":
    unittest.main()
