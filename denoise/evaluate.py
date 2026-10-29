"""Evaluate untouched test data with fixed, batch-independent noise."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import dataset, first_indices, fixed_noise, predict, scores, setup, sha256, split_indices, summarize, write_json
from .model import Denoiser


def gaussian_baseline(x):
    axis = torch.arange(-1, 2, device=x.device, dtype=x.dtype)
    weights = torch.exp(-axis.square() / (2 * 0.7 ** 2))
    kernel = weights[:, None] * weights[None, :]
    kernel = kernel / kernel.sum()
    patches = F.pad(x, (1, 1, 1, 1), mode="reflect").unfold(2, 3, 1).unfold(3, 3, 1)
    return (patches * kernel).sum((-1, -2))


@torch.inference_mode()
def evaluate(model, clean, noise, batch, device, ensemble):
    names = ["noisy", "gaussian", "single"] + (["ensemble"] if ensemble else [])
    predictions = {name: [] for name in names}
    for start in range(0, len(clean), batch):
        x = clean[start:start + batch].to(device)
        noisy = x + noise[start:start + batch].to(device)
        values = {"noisy": noisy, "gaussian": gaussian_baseline(noisy), "single": predict(model, noisy)}
        if ensemble:
            values["ensemble"] = predict(model, noisy, True)
        for name, value in values.items():
            predictions[name].append(value.cpu())
    predictions = {name: torch.cat(parts) for name, parts in predictions.items()}
    return predictions, {name: scores(value, clean) for name, value in predictions.items()}


def run(args):
    setup(2026)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = Denoiser(**cfg["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    images, labels = dataset(cfg["data_root"], args.split == "validation")
    indices = torch.arange(len(images))
    if args.split == "validation":
        if checkpoint["full"]:
            raise ValueError("A full-data model has no independent validation split")
        _, indices = split_indices(len(images), cfg["validation_size"], cfg["split_seed"])
        images, labels = images[indices], labels[indices]
    clean = images.float() / 255
    noise = fixed_noise(clean.shape, args.seed)
    predictions, metrics = evaluate(model, clean, noise, args.batch_size, device, args.ensemble)
    selected = first_indices(labels)
    method = "ensemble" if args.ensemble else "single"
    result = {"split": args.split, "count": len(clean), "noise_seed": args.seed, "sigma": 50 / 255,
              "checkpoint_sha256": sha256(args.checkpoint), "epoch": checkpoint["epoch"], "full_training": checkpoint["full"],
              "primary_method": method, "output_clipped": False, "metrics": {name: summarize(value) for name, value in metrics.items()},
              "first_per_digit": [{"digit": digit, "index": int(indices[i]), **{name: {key: float(v[i]) for key, v in values.items()} for name, values in metrics.items()}} for digit, i in enumerate(selected)]}
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "digit"] + [f"{name}_{key}" for name in metrics for key in ("l1", "mse", "psnr")])
        for i in range(len(clean)):
            writer.writerow([int(indices[i]), int(labels[i])] + [float(metrics[name][key][i]) for name in metrics for key in ("l1", "mse", "psnr")])
    np.savez_compressed(output / "digits.npz", clean=clean[selected].numpy(), indices=indices[selected].numpy(),
                        **{name: value[selected].numpy() for name, value in predictions.items()})
    if args.external:
        if args.split != "test":
            raise ValueError("External images are evaluated only after model selection")
        with Image.open(args.external) as image:
            external = torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255)[None, None]
        ext_noise = fixed_noise(external.shape, args.external_seed)
        values, ext_metrics = evaluate(model, external, ext_noise, 1, device, args.ensemble)
        result["external"] = {"input_sha256": sha256(args.external), "shape": list(external.shape[-2:]), "noise_seed": args.external_seed,
                              "noise_mean": float(ext_noise.mean()), "noise_std": float(ext_noise.std(unbiased=False)),
                              "metrics": {name: summarize(value) for name, value in ext_metrics.items()}}
        np.savez_compressed(output / "external.npz", clean=external.numpy(), **{name: value.numpy() for name, value in values.items()})
    result["noise_mean"] = float(noise.mean())
    result["noise_std"] = float(noise.std(unbiased=False))
    result["noisy_min"] = float(predictions["noisy"].min())
    result["noisy_max"] = float(predictions["noisy"].max())
    write_json(output / "metrics.json", result)
    for name, values in result["metrics"].items():
        print(f"{name}: L1={values['l1']:.6f} PSNR={values['psnr']:.4f} dB", flush=True)
    if "external" in result:
        for name, values in result["external"]["metrics"].items():
            print(f"external/{name}: L1={values['l1']:.6f} PSNR={values['psnr']:.4f} dB", flush=True)
    from .figures import render_results
    render_results(output, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/final/model.pt")
    parser.add_argument("--output", default="outputs/test")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--external")
    parser.add_argument("--external-seed", type=int, default=424242)
    run(parser.parse_args())
