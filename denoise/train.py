"""Train from scratch against unchanged original MNIST images."""

import argparse
import copy
import hashlib
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import (MODEL_VERSION, TARGET_POLICY, add_noise, check_checkpoint,
                     clean_batch, dataset, read_json, resolve_device,
                     save_checkpoint, setup, write_json)
from .model import Denoiser


def run(config, output, resume=None, device_name="auto"):
    cfg = read_json(config)
    if any(key in cfg for key in ("augment", "tone_probability", "tone_seed")):
        raise ValueError("Target-transform configurations are not supported")
    setup(cfg["seed"])
    device = resolve_device(device_name)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise ValueError("Run already exists; explicitly resume or choose a new output")
    originals, _ = dataset(cfg["data_root"], True)
    digest = hashlib.sha256(originals.numpy().tobytes()).hexdigest()
    images = originals.to(device)
    model = Denoiser(**cfg["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = device.type == "cuda" and cfg["amp"]
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype == torch.float16)
    history, start_epoch, step = [], 0, 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        check_checkpoint(state)
        if state["config"] != cfg or state["data_sha256"] != digest:
            raise ValueError("Resume configuration or original training data differs")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, start_epoch, step = state["history"], state["epoch"], state["step"]
    write_json(output / "config.json", cfg)
    protocol = {"target_policy": TARGET_POLICY, "model_version": MODEL_VERSION,
                "train_count": len(images), "data_sha256": digest, "target_transforms": [],
                "noise_mean": 0, "noise_sigma": 50 / 255, "noisy_input_clipped": False,
                "loss": "mean L1 against original image", "selection": "fixed final epoch",
                "parameters": sum(p.numel() for p in model.parameters())}
    write_json(output / "protocol.json", protocol)
    batches = math.ceil(len(images) / cfg["batch_size"])
    total_steps, warmup = cfg["epochs"] * batches, cfg["warmup_epochs"] * batches
    for epoch in range(start_epoch, cfg["epochs"]):
        begin = time.perf_counter()
        generator = torch.Generator(device=device).manual_seed(cfg["seed"] + epoch * 1009)
        order = torch.randperm(len(images), generator=generator, device=device)
        model.train()
        total_loss = 0.0
        for ids in order.split(cfg["batch_size"]):
            clean = clean_batch(images, ids)
            noisy = add_noise(clean, generator)
            step += 1
            ratio = step / max(1, warmup) if step <= warmup else cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))) / 2
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * ratio
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                prediction = model(noisy)
                loss = F.l1_loss(prediction.float(), clean, reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            decay = min(cfg["ema_decay"], (1 + step) / (10 + step))
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1 - decay)
            total_loss += float(loss.detach()) * len(ids)
        row = {"epoch": epoch + 1, "train_l1": total_loss / len(images),
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.perf_counter() - begin, 3)}
        history.append(row)
        payload = {"config": cfg, "model": ema.state_dict(), "epoch": epoch + 1,
                   "target_policy": TARGET_POLICY, "model_version": MODEL_VERSION, "data_sha256": digest}
        save_checkpoint(output / "model.pt", payload)
        save_checkpoint(output / "last.pt", {**payload, "model": model.state_dict(),
                        "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "step": step})
        write_json(output / "history.json", history)
        print(" ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--output", default="runs/original")
    parser.add_argument("--resume", help="Trusted checkpoint from the same original-target run")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args.config, args.output, args.resume, args.device)
