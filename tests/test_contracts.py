"""Numerical and structural contracts for the denoising experiment."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from denoise.common import SIGMA, add_noise, first_indices, fixed_noise, predict, read_json, resolve_device, save_checkpoint, scores, split_indices, structure_augment, structure_validation, tone_augment, write_json
from denoise.model import Denoiser, partition, reverse, shift_mask
from denoise.evaluate import gaussian_baseline
from denoise.train import run


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_assignment_implementation_budget(self):
        modules = list(Path("denoise").glob("*.py"))
        self.assertLessEqual(sum(len(path.read_text(encoding="utf-8").splitlines()) for path in modules), 800)

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

    def test_overlap_initialization_and_normalization(self):
        torch.manual_seed(16)
        base = Denoiser(dim=16, depth=2, heads=2, halo=2)
        nn.init.normal_(base.head.weight, std=0.01)
        model = Denoiser(dim=16, depth=2, heads=2, halo=2, output_halo=1)
        model.initialize(base.state_dict())
        self.assertFalse(any(isinstance(m, nn.modules.conv._ConvNd) for m in model.modules()))
        for shape in ((28, 28), (32, 32), (29, 31)):
            x = torch.randn(2, 1, *shape)
            torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            model.refine.bias.fill_(0.25)
        torch.testing.assert_close(model(x) - base(x), torch.full_like(x, 0.25))
        F.l1_loss(model(x), torch.zeros_like(x)).backward()
        self.assertGreater(float(model.refine.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(model.embed.weight.grad).all())
        broken = dict(base.state_dict())
        broken.pop("embed.weight")
        with self.assertRaises(ValueError):
            model.initialize(broken)

    def test_structure_targets_and_fresh_noise(self):
        clean = torch.rand(32, 1, 28, 28)
        original = clean.clone()
        torch.testing.assert_close(structure_augment(clean, 0, torch.Generator().manual_seed(3)), clean)
        changed = structure_augment(clean, 1, torch.Generator().manual_seed(3))
        torch.testing.assert_close(changed, structure_augment(clean, 1, torch.Generator().manual_seed(3)))
        torch.testing.assert_close(clean, original, rtol=0, atol=0)
        self.assertGreaterEqual(float(changed.min()), 0)
        self.assertLessEqual(float(changed.max()), 1)
        self.assertFalse(torch.allclose(changed, clean))
        noise = add_noise(changed, torch.Generator().manual_seed(11)) - changed
        self.assertAlmostEqual(float(noise.std()), SIGMA, delta=0.003)
        torch.testing.assert_close(noise, fixed_noise(changed.shape, 11))
        validation = structure_validation((clean * 255).byte(), 24)
        self.assertEqual(validation.shape, (32, 1, 32, 32))
        self.assertGreaterEqual(float(validation.min()), 0)
        self.assertLessEqual(float(validation.max()), 1)

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
        toned = tone_augment(clean[:32], 1, torch.Generator().manual_seed(4))
        self.assertGreaterEqual(float(toned.min()), 0)
        self.assertLessEqual(float(toned.max()), 1)

    def test_metrics_use_unit_peak_and_unclipped_values(self):
        clean = torch.zeros(2, 1, 28, 28)
        result = scores(torch.full_like(clean, -0.1), clean)
        torch.testing.assert_close(result["psnr"], torch.full((2,), 20.0, dtype=torch.float64))
        self.assertAlmostEqual(float(result["l1"].mean()), 0.1, places=6)
        self.assertTrue(torch.isinf(scores(clean, clean)["psnr"]).all())
        clean = torch.rand(5, 1, 28, 28)
        pred = clean + torch.randn_like(clean)
        self.assertTrue(torch.all(scores(pred.clamp(0, 1), clean)["l1"] <= scores(pred, clean)["l1"]))
        self.assertTrue(torch.all(scores(pred.clamp(0, 1), clean)["mse"] <= scores(pred, clean)["mse"]))

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

    def test_cpu_fallback_and_training_checkpoint(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")
        cfg = read_json("configs/overlap.json")
        cfg.update(epochs=2, validation_size=4, batch_size=4, eval_batch_size=4, tone_probability=0.5, tone_seed=6789,
                   structure_probability=0.5, structure_seed=2468, mixed_sizes=True)
        cfg["model"].update(dim=16, depth=2, heads=2, output_halo=1)
        images = torch.randint(0, 256, (16, 1, 28, 28), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as directory, patch("denoise.train.dataset", return_value=(images, None)):
            root = Path(directory)
            write_json(root / "config.json", cfg)
            def capture(path, value):
                save_checkpoint(path, value)
                if Path(path).name == "last.pt" and value["epoch"] == 1:
                    save_checkpoint(root / "epoch1.pt", value)
            with patch("denoise.train.save_checkpoint", side_effect=capture):
                history = run(root / "config.json", root / "selection", device_name="cpu")
            self.assertEqual(len(history), 2)
            saved = torch.load(root / "selection/model.pt", map_location="cpu", weights_only=True)
            self.assertFalse(saved["full"])
            self.assertIn("structure_l1", history[-1])
            expected = torch.load(root / "selection/last.pt", map_location="cpu", weights_only=False)["model"]
            restored = run(root / "config.json", root / "selection", resume=root / "epoch1.pt", device_name="cpu")
            for key in ("train_l1", "val_l1", "val_mse", "val_psnr"):
                self.assertEqual(history[-1][key], restored[-1][key])
            actual = torch.load(root / "selection/last.pt", map_location="cpu", weights_only=False)["model"]
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            full = run(root / "config.json", root / "full", full=True, device_name="cpu", initialize=root / "selection/model.pt")
            self.assertNotIn("val_l1", full[-1])
            with self.assertRaises(ValueError):
                run(root / "config.json", root / "invalid", device_name="cpu", initialize=root / "full/model.pt")


if __name__ == "__main__":
    unittest.main()
