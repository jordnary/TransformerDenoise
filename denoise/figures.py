"""Publication figures drawn from saved, unmodified floating-point arrays."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json


def comparison(arrays, method, path, columns, labels):
    fig, axes = plt.subplots(3, len(columns), figsize=(max(7, len(columns) * 1.9), 5.9), squeeze=False)
    for row, name in enumerate(("clean", "noisy", method)):
        for col, index in enumerate(columns):
            ax = axes[row, col]
            image = arrays[name][index, 0]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(labels[col], fontsize=11)
            else:
                mse = np.mean((image.astype(np.float64) - arrays["clean"][index, 0]) ** 2)
                ax.set_title(f"PSNR {(-10 * np.log10(mse)):.2f} dB", fontsize=10)
            if col == 0:
                ax.set_ylabel(("Clean", "Noisy", "Denoised")[row], fontsize=11)
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_results(output, method):
    output = Path(output)
    with np.load(output / "digits.npz") as arrays:
        for start in (0, 5):
            comparison(arrays, method, output / f"digits_{start}_{start + 4}.png", range(start, start + 5),
                       [f"Digit {i} | index {arrays['indices'][i]}" for i in range(start, start + 5)])
    if (output / "external.npz").exists():
        with np.load(output / "external.npz") as arrays:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
            h, w = arrays["clean"].shape[-2:]
            for ax, key, title in zip(axes, ("clean", "noisy", method), (f"Clean ({h} x {w})", "Noisy", "Denoised")):
                ax.imshow(arrays[key][0, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                if key != "clean":
                    mse = np.mean((arrays[key].astype(np.float64) - arrays["clean"]) ** 2)
                    title += f"\nPSNR {(-10 * np.log10(mse)):.2f} dB"
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(output / "external.png", dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def learning_curve(history_path, output):
    history = read_json(history_path)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].plot(epochs, [row["train_l1"] for row in history], color="#167d9a")
    axes[0].set(xlabel="Epoch", ylabel="Training L1 loss")
    axes[1].plot(epochs, [row["lr"] for row in history], color="#b65a28")
    axes[1].set(xlabel="Epoch", ylabel="Learning rate")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
