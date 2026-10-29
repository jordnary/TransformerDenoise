"""Data, deterministic noise, metrics, and inference shared by the commands."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

SIGMA = 50 / 255


def resolve_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or --device auto")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def setup(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def dataset(root, train):
    ds = MNIST(root, train=train, download=True)
    return ds.data[:, None], ds.targets


TARGET_POLICY = "original_uint8_div_255"
MODEL_VERSION = "centered_window_transformer_v1"


def clean_batch(images, indices):
    if images.dtype != torch.uint8:
        raise ValueError("Training targets must come from the original uint8 images")
    return images[indices].float() / 255


def check_checkpoint(checkpoint):
    if checkpoint.get("target_policy") != TARGET_POLICY or checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint does not match the original-target training protocol")


def add_noise(clean, generator=None):
    return clean + SIGMA * torch.randn(clean.shape, device=clean.device, generator=generator)


def fixed_noise(shape, seed):
    return SIGMA * torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def scores(pred, clean):
    error = (pred.double() - clean.double()).flatten(1)
    mse = error.square().mean(1)
    return {"l1": error.abs().mean(1), "mse": mse, "psnr": -10 * torch.log10(mse)}


def summarize(metrics):
    result = {key: float(value.mean()) for key, value in metrics.items()}
    psnr = metrics["psnr"]
    result["psnr_std"] = float(psnr.std(unbiased=len(psnr) > 1))
    result["psnr_standard_error"] = result["psnr_std"] / len(psnr) ** 0.5
    result["psnr_from_mse"] = float(-10 * torch.log10(metrics["mse"].mean()))
    return result


def first_indices(labels):
    return torch.tensor([int(torch.where(labels == digit)[0][0]) for digit in range(10)])


@torch.inference_mode()
def predict(model, noisy):
    return model(noisy).float()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
