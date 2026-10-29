"""Recalculate exported metrics and verify fixed noise without a model."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .common import fixed_noise, read_json, scores, summarize


def run(directory):
    root = Path(directory)
    result = read_json(root / "metrics.json")
    with (root / "per_image.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result["count"], "Image count mismatch"
    methods = list(result["metrics"])
    for method in methods:
        metrics = {key: torch.tensor([float(row[f"{method}_{key}"]) for row in rows], dtype=torch.float64) for key in ("l1", "mse", "psnr")}
        for key, value in summarize(metrics).items():
            np.testing.assert_allclose(value, result["metrics"][method][key], rtol=1e-10, atol=1e-10)
    with np.load(root / "digits.npz") as arrays:
        positions = {int(row["index"]): i for i, row in enumerate(rows)}
        selected = [positions[int(index)] for index in arrays["indices"]]
        clean = torch.from_numpy(arrays["clean"])
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
            noise = fixed_noise(clean.shape, result["external"]["noise_seed"])
            torch.testing.assert_close(torch.from_numpy(arrays["noisy"]), clean + noise, rtol=0, atol=0)
            for method in methods:
                for key, value in summarize(scores(torch.from_numpy(arrays[method]), clean)).items():
                    np.testing.assert_allclose(value, result["external"]["metrics"][method][key], rtol=1e-10, atol=1e-10)
    print(f"PASS: {len(rows)} image records, first-per-digit selection, fixed noise, and exported metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    run(parser.parse_args().directory)
