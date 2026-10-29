"""Verify original clean references, fixed noise, and exported metrics."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .common import TARGET_POLICY, dataset, first_indices, fixed_noise, read_json, scores, sha256, summarize


def run(directory, data_root="data", external=None):
    root = Path(directory)
    result = read_json(root / "metrics.json")
    assert result.get("target_policy") == TARGET_POLICY, "Incompatible target protocol"
    original, labels = dataset(data_root, result["split"] == "train-first")
    with (root / "per_image.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["count"], "Image count mismatch"
    assert all(int(row["digit"]) == int(labels[int(row["index"])]) for row in rows), "Source label mismatch"
    methods = list(result["metrics"])
    for method in methods:
        metrics = {key: torch.tensor([float(row[f"{method}_{key}"]) for row in rows], dtype=torch.float64) for key in ("l1", "mse", "psnr")}
        for key, value in summarize(metrics).items():
            np.testing.assert_allclose(value, result["metrics"][method][key], rtol=1e-10, atol=1e-10)
    with np.load(root / "digits.npz") as arrays:
        positions = {int(row["index"]): i for i, row in enumerate(rows)}
        selected = [positions[int(index)] for index in arrays["indices"]]
        clean = torch.from_numpy(arrays["clean"])
        indices = torch.from_numpy(arrays["indices"])
        torch.testing.assert_close(indices, first_indices(labels), rtol=0, atol=0)
        torch.testing.assert_close(clean, original[indices].float() / 255, rtol=0, atol=0)
        noise = fixed_noise((len(rows), *clean.shape[1:]), result["noise_seed"])[selected]
        torch.testing.assert_close(torch.from_numpy(arrays["noisy"]), clean + noise, rtol=0, atol=0)
        for digit, position in enumerate(selected):
            assert int(rows[position]["digit"]) == digit, "Digit mismatch"
            assert position == next(i for i, row in enumerate(rows) if int(row["digit"]) == digit), "Not the first sample"
        for method in methods:
            for key, value in scores(torch.from_numpy(arrays[method]), clean).items():
                expected = [float(rows[i][f"{method}_{key}"]) for i in selected]
                np.testing.assert_allclose(value.numpy(), expected, rtol=1e-10, atol=1e-10)
    if "external" in result:
        with np.load(root / "external.npz") as arrays:
            clean = torch.from_numpy(arrays["clean"])
            if external:
                with Image.open(external) as image:
                    expected = torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255)[None, None]
                torch.testing.assert_close(clean, expected, rtol=0, atol=0)
                assert sha256(external) == result["external"]["input_sha256"], "External file differs"
            noise = fixed_noise(clean.shape, result["external"]["noise_seed"])
            torch.testing.assert_close(torch.from_numpy(arrays["noisy"]), clean + noise, rtol=0, atol=0)
            for method in methods:
                for key, value in summarize(scores(torch.from_numpy(arrays[method]), clean)).items():
                    np.testing.assert_allclose(value, result["external"]["metrics"][method][key], rtol=1e-10, atol=1e-10)
    print(f"PASS: {len(rows)} records, original clean targets, first samples, noise, and metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--external", help="Original external image for pixel and digest verification")
    args = parser.parse_args()
    run(args.directory, args.data_root, args.external)
