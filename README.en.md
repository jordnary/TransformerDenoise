# Transformer Denoising

[简体中文](README.md) · **English**

A PyTorch experiment for grayscale image denoising. The model is trained from scratch using overlapping patches, windowed self-attention, two-dimensional relative position biases, and SwiGLU. It supports variable input sizes, contains no convolutional layers, and uses no existing denoising model or external pretrained weights.

## Method

- Observation model: `y = x + ε`, with `x ∈ [0, 1]` and `ε ~ N(0, (50/255)²)`. Noisy inputs are not clipped.
- Objective: mean pixelwise L1. PSNR uses a fixed peak of `1`; mean per-image PSNR and PSNR derived from aggregate MSE are reported separately.
- Architecture: linear projections of `6 × 6` patches at stride `2`, eight regular/shifted-window Transformer blocks, 64 feature channels, four attention heads, and 338,660 parameters.
- Spatial dimensions: reflection padding, linear projections, and cropping support `28 × 28` and `32 × 32` without resizing.
- Training: learn digit structure first, then broaden the distribution using brightness, contrast, and grayscale inversion. Original and intensity-varied validation L1 are weighted equally for selection.
- Default inference: one forward pass and projection to `[0, 1]`. Eight-transform self-ensembling is an optional comparison.

## Installation and training

Python 3.11 or later is required. Install a device-compatible PyTorch build following the [official instructions](https://pytorch.org/get-started/locally/), then install the project dependencies.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denoise.train --config configs/overlap.json --output runs/overlap
python -m denoise.train --config configs/robust.json --output runs/robust --initialize runs/overlap/model.pt
python -m denoise.train --config configs/robust_final.json --output runs/robust_final --full --initialize runs/robust/model.pt
```

MNIST is downloaded to `data/`. The first two stages use a fixed 55,000/5,000 training/validation split. The final stage optimizes all 60,000 training images for a fixed budget. `model.pt` stores EMA weights; `last.pt` stores resumable state. Full-data models cannot be used for independent validation selection; the program rejects such evaluation or initialization.

```powershell
python -m denoise.train --config configs/robust.json --output runs/robust --resume runs/robust/last.pt
python -m denoise.evaluate --checkpoint runs/robust/model.pt --split validation --seed 12345 --ensemble --project-output --output outputs/validation
```

`--initialize` transfers weights between stages. `--resume` restores the same stage and requires a matching configuration. A new run never silently overwrites existing training checkpoints.

## Evaluation and verification

```powershell
python -m denoise.evaluate --checkpoint runs/robust_final/model.pt --project-output --output outputs/test
python -m denoise.audit outputs/test
python -m denoise.evaluate --checkpoint runs/robust_final/model.pt --project-output --external personal/image.png --output outputs/example
```

`personal/image.png` denotes a user-provided local image that is not distributed with the repository. External images are converted to grayscale without resizing; evaluation reads images without updating weights. The default is the official 10,000-image MNIST test set. Digit examples are the first sample of each class in the original order. `--split train-first` evaluates first training samples and explicitly marks the output as training data.

Noise is generated before batched inference and does not depend on batch size. Outputs include per-image CSV metrics, JSON summaries, floating-point NPZ files, and figures. `--project-output` constrains only the reconstruction; raw-output metrics remain available. The fixed `[0, 1]` display range does not modify metric arrays.

Baselines include the noisy input and a fixed `3 × 3` Gaussian smoother. `--ensemble` enables eight-transform self-ensembling, requiring eight forward passes. MNIST and natural textures differ in distribution; intensity augmentation does not establish reliable generalization to all natural images.

## Structure augmentation and overlapping refinement

An additional controlled experiment compares continued training, MNIST structure augmentation, and structure augmentation with an overlapping refinement branch. Clean affine crops and convex mixtures use only MNIST images within the current split; fresh fixed-strength Gaussian noise is added afterwards. Mixed-size training covers `28 × 28` and `32 × 32`. The optional branch combines shallow and deep features, predicts overlapping residual patches through a linear layer, and normalizes their sums by coverage counts. It uses no convolution.

```powershell
python -m denoise.experiment --external personal/image.png
```

Complete the base and intensity-augmentation stages above first. The experiment selects by a fixed three-domain validation L1 criterion with a maximum 0.20 dB digit-validation PSNR loss, then trains on all 60,000 images for eight fixed epochs before reading the external test image. Outputs include validation comparisons, the selection decision, paired old/new evaluations across three fixed noise seeds, and a five-panel image comparison.

Comparison configurations are in `configs/experiment_*.json`, generated results go to `outputs/experiment/`, and checkpoints go to `runs/experiment/`. Repeating the command resumes training stages; retain their configurations and initialization checkpoints. Personal experiment plans, reports, and metric summaries belong in local `reports/experiments/` and are not distributed with the repository.

## CPU fallback

`--device auto` selects an available backend; `--device cpu` forces CPU execution. CPU uses full precision without gradient scaling and supports training, checkpoint resumption, and inference from the same checkpoint.

```powershell
python -m denoise.train --config configs/cpu.json --device cpu --output runs/cpu
python -m denoise.evaluate --checkpoint runs/robust_final/model.pt --device cpu --batch-size 16 --project-output --output outputs/cpu
```

The CPU configuration offers a separate recipe with smaller batches and the same architecture. Add `--device cpu` to the corresponding commands to execute the complete three-stage recipe. Different configurations, devices, and backends do not guarantee bitwise agreement; small CPU checks do not constitute a complete training replication.

## Repository layout and local files

```text
README.md            Chinese README
README.en.md         English README
requirements.txt     Python dependencies
configs/             Training and comparison configurations
denoise/
  model.py           Linear patch projections and Transformer blocks
  common.py          Data, augmentation, noise, metrics, inference
  train.py           Optimization, selection, and checkpointing
  evaluate.py        Quantitative evaluation and exports
  figures.py         Figures from floating-point arrays
  audit.py           Independent verification of exported metrics
  experiment.py      Controlled comparison driver
tests/               Structural and numerical contract tests
docs/                Public project documentation
assignment/          Local assignment requirements and supplied materials
reports/             Local reports, experiment records, and submission drafts
personal/            Local notes, input images, and archives
data/                Downloaded datasets (generated locally)
runs/                Checkpoints and training state (generated locally)
outputs/             Evaluation exports and figures (generated locally)
```

Both READMEs stay in the repository root, with language-switch links at the top. Use English ASCII file and directory names, preferably `snake_case`.

Use `assignment/` for assignment requirements and supplied materials, `reports/` for reports, experiment plans, metric summaries, and submission drafts, and `personal/` for personal files. Older backups belong in `personal/archive/`. Only empty `.gitkeep` placeholders are tracked in these three directories; `.gitignore` excludes their actual contents. Keep only general project documentation in `docs/` and do not link local reports from the READMEs.

`nonoverlap.json`, `context.json`, and `final.json` retain architecture and non-intensity-augmented comparison configurations. Data, weights, run records, and generated outputs are also excluded by `.gitignore`; their directories are created as needed at runtime. Only load complete training checkpoints from trusted sources. See the [repository layout guide](docs/repository_layout.md) for directory responsibilities and pre-commit checks.

## References

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Liu et al. [Swin Transformer](https://arxiv.org/abs/2103.14030).
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
