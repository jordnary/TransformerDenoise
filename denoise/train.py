"""Train with fresh Gaussian corruption and validation-only model selection."""

import argparse
import copy
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import add_noise, dataset, fixed_noise, read_json, setup, split_indices, write_json
from .model import Denoiser


@torch.inference_mode()
def validate(model, images, noise, batch_size, device):
    model.eval()
    errors = torch.zeros(3, device=device, dtype=torch.float64)
    for start in range(0, len(images), batch_size):
        clean = images[start:start + batch_size].to(device).float() / 255
        noisy = clean + noise[start:start + batch_size].to(device)
        pred = model(noisy).float()
        diff = (pred - clean).double().flatten(1)
        mse = diff.square().mean(1)
        errors += torch.stack((diff.abs().mean(1).sum(), mse.sum(), (-10 * mse.log10()).sum()))
    return dict(zip(("l1", "mse", "psnr"), (errors / len(images)).tolist()))


def run(config, output, full=False, resume=None):
    cfg = read_json(config)
    setup(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise ValueError("Run already exists; choose another output or explicitly resume")
    all_images, _ = dataset(cfg["data_root"], True)
    train_ids, val_ids = split_indices(len(all_images), cfg["validation_size"], cfg["split_seed"])
    images = all_images if full else all_images[train_ids]
    validation = None if full else all_images[val_ids]
    noise = None if full else fixed_noise(validation.shape, cfg["validation_noise_seed"])
    images = images.to(device)
    model = Denoiser(**cfg["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = device.type == "cuda" and cfg["amp"]
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype == torch.float16)
    history, best, start_epoch, step = [], float("inf"), 0, 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["config"] != cfg or state["full"] != full:
            raise ValueError("Resume configuration differs from the saved run")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best, start_epoch, step = state["history"], state["best"], state["epoch"], state["step"]
    write_json(output / "config.json", cfg)
    write_json(output / "protocol.json", {"full": full, "train_count": len(images), "validation_count": 0 if full else len(validation), "parameters": sum(p.numel() for p in model.parameters()), "selection": "fixed final epoch" if full else "minimum fixed-noise validation L1", "train_indices": None if full else train_ids.tolist(), "validation_indices": None if full else val_ids.tolist()})
    batches = math.ceil(len(images) / cfg["batch_size"])
    total_steps, warmup = cfg["epochs"] * batches, cfg["warmup_epochs"] * batches
    for epoch in range(start_epoch, cfg["epochs"]):
        begin = time.perf_counter()
        generator = torch.Generator(device=device).manual_seed(cfg["seed"] + epoch * 1009)
        order = torch.randperm(len(images), generator=generator, device=device)
        model.train()
        total_loss = 0.0
        for ids in order.split(cfg["batch_size"]):
            clean = images[ids].float() / 255
            if cfg["augment"]:
                code = int(torch.randint(8, (), generator=generator, device=device))
                clean = clean.rot90(code % 4, (-2, -1))
                clean = clean.flip(-1) if code >= 4 else clean
            noisy = add_noise(clean, generator)
            step += 1
            ratio = step / max(1, warmup) if step <= warmup else cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))) / 2
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * ratio
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                pred = model(noisy)
                loss = F.l1_loss(pred.float(), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            decay = min(cfg["ema_decay"], (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
            total_loss += float(loss.detach()) * len(ids)
        row = {"epoch": epoch + 1, "train_l1": total_loss / len(images), "lr": optimizer.param_groups[0]["lr"]}
        if not full:
            row.update({"val_" + key: value for key, value in validate(ema, validation, noise, cfg["eval_batch_size"], device).items()})
        row["seconds"] = round(time.perf_counter() - begin, 3)
        history.append(row)
        improved = full or row["val_l1"] < best
        best = best if full else min(best, row["val_l1"])
        payload = {"config": cfg, "model": ema.state_dict(), "epoch": epoch + 1, "full": full}
        if improved:
            torch.save(payload, output / "model.pt")
        torch.save({**payload, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history, "best": best, "step": step}, output / "last.pt")
        write_json(output / "history.json", history)
        print(" ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--output", default="runs/selection")
    parser.add_argument("--full", action="store_true", help="Use all training images, with a fixed epoch budget")
    parser.add_argument("--resume", help="Trusted local training checkpoint")
    args = parser.parse_args()
    run(args.config, args.output, args.full, args.resume)
