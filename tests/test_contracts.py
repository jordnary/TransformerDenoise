"""Contracts for original targets, architecture, noise, and CPU execution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from denoise.common import (SIGMA, TARGET_POLICY, add_noise, check_checkpoint,
                           clean_batch, first_indices, fixed_noise, predict,
                           read_json, resolve_device, save_checkpoint, scores, write_json)
from denoise.model import Denoiser, partition, reverse, shift_mask
from denoise.train import run


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
            self.assertEqual(model(x).shape, x.shape)
            torch.testing.assert_close(model(x), x)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 3, 28, 28))

    def test_brightness_equivariance_and_batch_independence(self):
        torch.manual_seed(8)
        model = Denoiser(dim=16, depth=2, heads=2).eval()
        nn.init.normal_(model.head.weight, std=0.02)
        x = torch.randn(3, 1, 28, 32)
        y = predict(model, x)
        torch.testing.assert_close(predict(model, x + 0.4), y + 0.4, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(y, torch.cat([predict(model, b) for b in x.split(1)]), atol=2e-6, rtol=2e-6)

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

    def test_metrics_and_output_projection(self):
        clean = torch.zeros(2, 1, 28, 28)
        result = scores(torch.full_like(clean, -0.1), clean)
        torch.testing.assert_close(result["psnr"], torch.full((2,), 20.0, dtype=torch.float64))
        self.assertTrue(torch.isinf(scores(clean, clean)["psnr"]).all())
        clean = torch.rand(5, 1, 28, 28)
        pred = clean + torch.randn_like(clean)
        self.assertTrue(torch.all(scores(pred.clamp(0, 1), clean)["l1"] <= scores(pred, clean)["l1"]))
        self.assertTrue(torch.all(scores(pred.clamp(0, 1), clean)["mse"] <= scores(pred, clean)["mse"]))

    def test_original_targets_and_first_occurrence(self):
        images = torch.randint(0, 256, (20, 1, 28, 28), dtype=torch.uint8)
        before = images.clone()
        indices = torch.tensor([8, 2, 15, 3])
        target = clean_batch(images, indices)
        torch.testing.assert_close(target, before[indices].float() / 255, rtol=0, atol=0)
        torch.testing.assert_close(images, before, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            clean_batch(target, torch.arange(len(target)))
        labels = torch.tensor([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8])
        self.assertEqual(first_indices(labels).tolist(), [1, 3, 5, 7, 2, 0, 13, 15, 17, 4])
        with self.assertRaises(ValueError):
            check_checkpoint({"config": {"tone_probability": 0.5}})

    def test_cpu_training_gt_and_exact_resume(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
        cfg = read_json("configs/train.json")
        cfg.update(epochs=2, batch_size=4)
        cfg["model"].update(dim=16, depth=2, heads=2)
        images = torch.randint(0, 256, (16, 1, 28, 28), dtype=torch.uint8)
        reference = images.float() / 255
        real_l1 = F.l1_loss
        seen = []
        def inspect_loss(pred, target, **kwargs):
            for item in target:
                self.assertTrue(any(torch.equal(item, original) for original in reference))
                seen.append(item.clone())
            return real_l1(pred, target, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch("denoise.train.dataset", return_value=(images, None)):
            root = Path(directory)
            write_json(root / "config.json", cfg)
            def capture(path, value):
                save_checkpoint(path, value)
                if Path(path).name == "last.pt" and value["epoch"] == 1:
                    save_checkpoint(root / "epoch1.pt", value)
            with patch("denoise.train.save_checkpoint", side_effect=capture), patch("denoise.train.F.l1_loss", side_effect=inspect_loss):
                history = run(root / "config.json", root / "run", device_name="cpu")
            self.assertEqual(len(seen), 32)
            expected = torch.load(root / "run/last.pt", map_location="cpu", weights_only=False)["model"]
            restored = run(root / "config.json", root / "run", resume=root / "epoch1.pt", device_name="cpu")
            self.assertEqual(history[-1]["train_l1"], restored[-1]["train_l1"])
            actual = torch.load(root / "run/last.pt", map_location="cpu", weights_only=False)["model"]
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            saved = torch.load(root / "run/model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["target_policy"], TARGET_POLICY)


if __name__ == "__main__":
    unittest.main()
