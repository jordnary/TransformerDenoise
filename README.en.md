# Transformer Denoising

[简体中文](README.md) · **English**

A grayscale denoising model trained from scratch in PyTorch. It uses overlapping patches, windowed self-attention, two-dimensional relative position biases, and SwiGLU. There are no convolutional layers, existing denoising models, or external pretrained weights.

## Model and data

- Original MNIST pixels are divided only by 255. Targets remain the original clean images without rotation, reflection, cropping, or intensity changes.
- Independent zero-mean Gaussian noise with standard deviation 50/255 is added without clipping noisy inputs.
- Relative intensities are computed inside the network; an input residual preserves absolute brightness. Feature computation does not modify targets.
- Eight window Transformer blocks, 64 feature channels, four attention heads, and 338,660 parameters.
- The same model directly handles 28×28, 32×32, and other supported sizes, preserving input dimensions.
- Training uses only mean L1. Final reconstructions are projected to [0,1], with raw-output metrics retained.

## Training

Python 3.11 or later is required. Install a backend-appropriate PyTorch build, then install the dependencies.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denoise.train --config configs/train.json --output runs/original
```

All 60,000 original training images are used for a fixed epoch budget; the official test set is excluded from training. Data is downloaded to `data/`. `model.pt` stores EMA weights, and `last.pt` stores complete resumable state. The fixed final-epoch checkpoint is evaluated.

```powershell
python -m denoise.train --config configs/train.json --output runs/original --resume runs/original/last.pt
```

Resumption requires matching configuration, original-data digest, and target protocol. Incompatible legacy checkpoints are rejected, and existing training is not silently overwritten. Only load complete training checkpoints from trusted sources.

## Evaluation and verification

```powershell
python -m denoise.evaluate --checkpoint runs/original/model.pt --split train-first --external examples/image.png --output outputs/required
python -m denoise.evaluate --checkpoint runs/original/model.pt --split test --seed 20261029 --output outputs/test
python -m denoise.audit outputs/required
python -m denoise.audit outputs/test
```

`examples/image.png` denotes a user-provided grayscale image. Its dimensions are preserved and noise is added only to a copy. `train-first` selects the first sample of each class in the original training order; `test` evaluates all 10,000 official test images. These results are clearly distinguished.

Noise is generated before batching and does not depend on batch size. Outputs include comparison figures, per-image CSV metrics, JSON summaries, and floating-point NPZ files. PSNR uses a fixed peak of 1; mean per-image PSNR and PSNR derived from aggregate MSE are both reported. The [0,1] display range does not modify noisy evaluation arrays.

## CPU fallback

`--device auto` selects an available backend; `--device cpu` forces CPU execution. CPU supports full training, exact resumption in the same environment, and inference from the same weights without mixed precision.

```powershell
python -m denoise.train --config configs/cpu.json --device cpu --output runs/cpu
python -m denoise.evaluate --checkpoint runs/original/model.pt --device cpu --batch-size 16 --split test --output outputs/cpu
```

The CPU configuration uses smaller batches. To retain the main experiment's hyperparameters, specify `--device cpu` with `configs/train.json`. Minor floating-point differences across compute backends are possible.

## Layout

```text
configs/             Training and CPU configurations
denoise/
  model.py           Linear patch projections and attention blocks
  common.py          Original targets, noise, metrics and checkpoints
  train.py           L1 training and checkpoint resumption
  evaluate.py        Fixed-sample and full-test evaluation
  figures.py         Comparison figures and training curves
  audit.py           Independent verification of numerical exports
tests/               Data, architecture and numerical contract tests
```

Data, weights, run records, and generated results are excluded by `.gitignore`. Natural images differ from MNIST; a single reconstruction does not establish general natural-image denoising performance.
