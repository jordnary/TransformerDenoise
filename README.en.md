# Transformer Denoising

[简体中文](README.md) · **English**

A PyTorch experiment for grayscale image denoising. The model is trained from scratch and combines linear patch projections, windowed self-attention, two-dimensional relative position biases, and SwiGLU. It contains no convolutional layers and uses no pretrained weights.

## Method

- Corruption: `y = x + ε`, with `x ∈ [0, 1]` and `ε ~ N(0, (50/255)²)`. No clipping is applied to noisy inputs.
- Objective: mean pixelwise L1 loss. PSNR uses a fixed peak value of `1`; both mean per-image PSNR and PSNR derived from aggregate MSE are reported.
- Architecture: linear projections of overlapping `6 × 6` patches at stride `2`, eight alternating regular/shifted-window Transformer blocks, four attention heads, 64 feature channels, and residual reconstruction.
- Spatial dimensions: reflection padding to the window grid followed by cropping to the original dimensions; supports `28 × 28`, `32 × 32`, and other grayscale sizes above the minimum supported dimensions.
- Selection: a fixed 55,000/5,000 training/validation split, fresh training noise, fixed validation noise, and checkpoint selection by validation L1.
- Inference: a single forward pass or self-ensembling over eight rotations/reflections. Metrics are retained for both raw predictions and outputs projected to `[0, 1]`.

## Installation and training

Python 3.10 or later is required. Install a device-compatible PyTorch build following the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denoise.train --config configs/overlap.json --output runs/selection
```

MNIST is downloaded to `data/`. JSON files define the experiment parameters. `runs/selection/model.pt` contains the exponential-moving-average weights with the lowest validation L1; `last.pt` contains the complete state for resuming training.

```powershell
python -m denoise.train --config configs/overlap.json --output runs/selection --resume runs/selection/last.pt
python -m denoise.evaluate --checkpoint runs/selection/model.pt --split validation --seed 12345 --ensemble --project-output --output outputs/validation
```

Once the final training budget is fixed, use `--full` to train on all 60,000 training images. A full-data model no longer has an independent validation set; the evaluator rejects validation evaluation for such a checkpoint.

```powershell
python -m denoise.train --config configs/overlap.json --output runs/full --full
python -m denoise.evaluate --checkpoint runs/full/model.pt --ensemble --project-output --output outputs/test
```

## Evaluation

The test set is evaluated in its original order. The first sample of each digit is selected by its label index. Noise is generated before batched inference and is independent of inference batch size. Training, validation, and test seeds are separate.

```powershell
python -m denoise.evaluate --checkpoint runs/full/model.pt --ensemble --project-output --external examples/image.png --output outputs/example
python -m denoise.audit outputs/example
```

`--external` accepts a user-provided image and converts it to grayscale without resizing. External images are excluded from training and selection. Outputs include per-image CSV metrics, JSON summaries, NPZ files preserving floating-point pixels, and comparison figures. `--project-output` constrains only the final reconstruction to the valid intensity range; raw-output metrics remain available. The fixed display range `[0, 1]` does not modify metric arrays.

Baselines are the noisy input and a fixed `3 × 3` Gaussian smoother. Self-ensembling requires eight forward passes and should be selected using validation data. MNIST represents handwritten digits; performance on natural images must be assessed separately.

## CPU execution

`--device auto` selects an available compute backend. `--device cpu` forces CPU execution without code changes. CPU execution uses full precision and disables mixed precision and gradient scaling. The CPU configuration reduces batch size while keeping the architecture unchanged.

```powershell
python -m denoise.train --config configs/cpu.json --device cpu --output runs/cpu
python -m denoise.evaluate --checkpoint runs/full/model.pt --device cpu --batch-size 16 --ensemble --project-output --output outputs/cpu
```

Configuration, random noise, and optimization updates affect results. The CPU configuration is a separate training recipe; a partial CPU run does not reproduce a fully trained experiment from another configuration.

## Layout

```text
configs/             Experiment configurations
denoise/
  model.py           Attention blocks and residual reconstruction
  common.py          Data, noise, metrics, and inference
  train.py           Optimization, validation, and checkpointing
  evaluate.py        Evaluation and numerical exports
  figures.py         Figures from saved floating-point arrays
  audit.py           Independent verification of exported results
tests/               Numerical and architectural contract tests
```

Datasets, weights, run records, and generated outputs are excluded by `.gitignore`. Only load trusted training checkpoints: resumable states include optimizer objects. Fixed seeds support reproducibility, but bitwise agreement across devices or PyTorch backends is not guaranteed.

## References

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Liu et al. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

The network components are implemented directly in this project, using published architectural ideas.
