"""Evaluate original clean images with fixed Gaussian corruption."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .common import (check_checkpoint, dataset, first_indices, fixed_noise, predict,
                     resolve_device, scores, setup, sha256, summarize, write_json)
from .model import Denoiser


@torch.inference_mode()
def evaluate(model, clean, noise, batch, device):
    predictions = {"noisy": [], "raw": []}
    for start in range(0, len(clean), batch):
        target = clean[start:start + batch].to(device)
        noisy = target + noise[start:start + batch].to(device)
        predictions["noisy"].append(noisy.cpu())
        predictions["raw"].append(predict(model, noisy).cpu())
    predictions = {name: torch.cat(parts) for name, parts in predictions.items()}
    predictions["denoised"] = predictions["raw"].clamp(0, 1)
    return predictions, {name: scores(value, clean) for name, value in predictions.items()}


def run(args):
    setup(2026)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    check_checkpoint(checkpoint)
    model = Denoiser(**checkpoint["config"]["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    images, labels = dataset(checkpoint["config"]["data_root"], args.split == "train-first")
    indices = torch.arange(len(images))
    if args.split == "train-first":
        indices = first_indices(labels)
        images, labels = images[indices], labels[indices]
    clean = images.float() / 255
    noise = fixed_noise(clean.shape, args.seed)
    predictions, metrics = evaluate(model, clean, noise, args.batch_size, device)
    selected = first_indices(labels)
    result = {"split": args.split, "count": len(clean), "noise_seed": args.seed, "sigma": 50 / 255,
              "checkpoint_sha256": sha256(args.checkpoint), "epoch": checkpoint["epoch"],
              "target_policy": checkpoint["target_policy"], "model_version": checkpoint["model_version"],
              "primary_method": "denoised", "output_clipped": True, "input_clipped": False,
              "metrics": {name: summarize(value) for name, value in metrics.items()},
              "first_per_digit": [{"digit": digit, "index": int(indices[i]),
                                  **{name: {key: float(v[i]) for key, v in values.items()}
                                     for name, values in metrics.items()}} for digit, i in enumerate(selected)]}
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "digit"] + [f"{name}_{key}" for name in metrics for key in ("l1", "mse", "psnr")])
        for i in range(len(clean)):
            writer.writerow([int(indices[i]), int(labels[i])] + [float(metrics[name][key][i]) for name in metrics for key in ("l1", "mse", "psnr")])
    np.savez_compressed(output / "digits.npz", clean=clean[selected].numpy(), indices=indices[selected].numpy(),
                        **{name: value[selected].numpy() for name, value in predictions.items()})
    if args.external:
        with Image.open(args.external) as image:
            external = torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255)[None, None]
        ext_noise = fixed_noise(external.shape, args.external_seed)
        values, ext_metrics = evaluate(model, external, ext_noise, 1, device)
        result["external"] = {"input_sha256": sha256(args.external), "shape": list(external.shape[-2:]),
                              "noise_seed": args.external_seed, "noise_mean": float(ext_noise.mean()),
                              "noise_std": float(ext_noise.std(unbiased=False)),
                              "metrics": {name: summarize(value) for name, value in ext_metrics.items()}}
        np.savez_compressed(output / "external.npz", clean=external.numpy(), **{name: value.numpy() for name, value in values.items()})
    result.update(noise_mean=float(noise.mean()), noise_std=float(noise.std(unbiased=False)),
                  noisy_min=float(predictions["noisy"].min()), noisy_max=float(predictions["noisy"].max()))
    write_json(output / "metrics.json", result)
    for name, values in result["metrics"].items():
        print(f"{name}: L1={values['l1']:.6f} PSNR={values['psnr']:.4f} dB", flush=True)
    if "external" in result:
        for name, values in result["external"]["metrics"].items():
            print(f"external/{name}: L1={values['l1']:.6f} PSNR={values['psnr']:.4f} dB", flush=True)
    from .figures import render_results
    render_results(output, "denoised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/original/model.pt")
    parser.add_argument("--output", default="outputs/required")
    parser.add_argument("--split", choices=("test", "train-first"), default="train-first")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--external")
    parser.add_argument("--external-seed", type=int, default=424242)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())
