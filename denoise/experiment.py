"""Run preregistered MNIST-only comparisons, then evaluate the frozen choice."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .audit import run as audit
from .common import dataset, fixed_noise, read_json, resolve_device, setup, sha256, split_indices, structure_validation, tone_augment, write_json
from .evaluate import run as evaluate
from .model import Denoiser
from .train import run as train, validate


def run(args):
    setup(2030)
    root, output = Path(args.runs), Path(args.output)
    source = torch.load(args.initialize, map_location="cpu", weights_only=True)
    if source["full"]:
        raise ValueError("Comparison initialization must exclude validation images")
    cfg, device = source["config"], resolve_device(args.device)
    images, _ = dataset(cfg["data_root"], True)
    _, ids = split_indices(len(images), cfg["validation_size"], cfg["split_seed"])
    clean = images[ids]
    domains = {"val": clean, "tone": tone_augment(clean.float() / 255, 1, torch.Generator().manual_seed(6789)),
               "structure": structure_validation(clean, 24680)}
    model = Denoiser(**cfg["model"]).to(device).eval()
    model.load_state_dict(source["model"])
    baseline = {"arm": "initial", "epoch": source["epoch"]}
    for name, target in domains.items():
        values = validate(model, target, fixed_noise(target.shape, 12345), 256, device)
        baseline.update({name + "_" + key: value for key, value in values.items()})
    baseline["selection_l1"] = sum(baseline[name + "_l1"] for name in domains) / len(domains)
    del model, domains
    rows = [baseline]
    for arm in ("control", "structure", "refinement"):
        folder = root / arm
        last = folder / "last.pt"
        history = train(f"configs/experiment_{arm}.json", folder, resume=last if last.exists() else None,
                        device_name=args.device, initialize=None if last.exists() else args.initialize)
        best = min(history, key=lambda row: row["selection_l1"])
        rows.append({"arm": arm, **best})
        write_json(output / "validation.json", rows)
    eligible = [row for row in rows if row["val_psnr"] >= baseline["val_psnr"] - 0.20]
    winner = min(eligible, key=lambda row: row["selection_l1"])
    selected = Path(args.initialize) if winner["arm"] == "initial" else root / winner["arm"] / "model.pt"
    selected_cfg = torch.load(selected, map_location="cpu", weights_only=True)["config"]
    final_cfg = {**selected_cfg, "seed": 2031, "epochs": 8, "lr": 5e-5, "min_lr_ratio": 0.2}
    write_json(root / "final_config.json", final_cfg)
    decision = {"selected": winner["arm"], "selection": "minimum three-domain validation L1 with at most 0.20 dB digit loss",
                "validation": rows, "initialization_sha256": sha256(args.initialize), "selected_sha256": sha256(selected),
                "inference": "single_projected", "external_seeds": [424242, 424243, 424244]}
    write_json(output / "decision.json", decision)
    print(f"SELECTED: {winner['arm']} (validation only)", flush=True)
    last = root / "final" / "last.pt"
    train(root / "final_config.json", root / "final", full=True, resume=last if last.exists() else None,
          device_name=args.device, initialize=None if last.exists() else selected)
    for label, checkpoint in (("previous", args.previous), ("candidate", root / "final" / "model.pt")):
        for i in range(3):
            directory = output / f"{label}_{i + 1}"
            options = SimpleNamespace(checkpoint=checkpoint, output=directory, device=args.device, split="test",
                                      seed=20261029 + i, external_seed=424242 + i, external=args.external,
                                      batch_size=128, ensemble=False, project_output=True)
            evaluate(options)
            audit(directory)
    options.checkpoint, options.output, options.split, options.external = root / "final" / "model.pt", output / "train_first", "train-first", None
    options.seed = 20261101
    evaluate(options)
    audit(options.output)
    import matplotlib.pyplot as plt
    with np.load(output / "previous_1" / "external.npz") as old, np.load(output / "candidate_1" / "external.npz") as new:
        np.testing.assert_array_equal(old["clean"], new["clean"])
        np.testing.assert_array_equal(old["noisy"], new["noisy"])
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
        panels = [("Clean", old["clean"]), ("Noisy", old["noisy"]), ("Previous", old["single_projected"]),
                  ("Candidate", new["single_projected"]), ("Gaussian baseline", old["gaussian_projected"])]
        for ax, (title, pixels) in zip(axes, panels):
            ax.imshow(pixels[0, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if title != "Clean":
                title += f"\nPSNR {-10 * np.log10(np.mean((pixels.astype(np.float64) - old['clean']) ** 2)):.2f} dB"
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", default="runs/robust/model.pt")
    parser.add_argument("--previous", default="runs/robust_final/model.pt")
    parser.add_argument("--runs", default="runs/experiment")
    parser.add_argument("--output", default="outputs/experiment")
    parser.add_argument("--external", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())
