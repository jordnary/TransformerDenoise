# Transformer Denoising

**简体中文** · [English](README.en.md)

基于 PyTorch 的灰度图像去噪实验。模型从零训练，通过线性图像块映射、窗口自注意力、二维相对位置偏置与 SwiGLU 重建干净图像，网络不含卷积层，也不使用预训练权重。

## 方法

- 噪声模型：`y = x + ε`，其中 `x ∈ [0, 1]`，`ε ~ N(0, (50/255)²)`；加噪输入不裁剪。
- 重建目标：像素平均 L1 损失；PSNR 使用固定峰值 `1`，同时报告平均逐图 PSNR 与由全体 MSE 换算的 PSNR。
- 模型：步长为 `2` 的 `6 × 6` 重叠图像块线性映射、8 个交替普通/移位窗口 Transformer 块、4 个注意力头、64 维特征、残差输出。
- 尺寸处理：反射填充至窗口网格，推理后裁回输入尺寸；支持 `28 × 28`、`32 × 32` 及其他满足最小尺寸的灰度输入。
- 模型选择：固定的 55,000/5,000 训练/验证划分，动态训练噪声与固定验证噪声，依据验证 L1 保存权重。
- 推理：单次前向或八种旋转/镜像变换的自集成；分别记录原始输出与投影至 `[0, 1]` 后的指标。

## 安装与训练

需要 Python 3.10 或更新版本。按照 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装与计算设备匹配的 PyTorch，再安装其余依赖。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denoise.train --config configs/overlap.json --output runs/selection
```

MNIST 会下载到 `data/`。训练参数集中在 JSON 配置中。`runs/selection/model.pt` 保存验证 L1 最优的指数移动平均权重，`last.pt` 保存完整的断点续训状态。

```powershell
python -m denoise.train --config configs/overlap.json --output runs/selection --resume runs/selection/last.pt
python -m denoise.evaluate --checkpoint runs/selection/model.pt --split validation --seed 12345 --ensemble --project-output --output outputs/validation
```

固定最终训练预算后，使用 `--full` 在全部 60,000 张训练图像上训练。完整训练模型不再具有独立验证集；`evaluate` 会拒绝对其运行验证集评估。

```powershell
python -m denoise.train --config configs/overlap.json --output runs/full --full
python -m denoise.evaluate --checkpoint runs/full/model.pt --ensemble --project-output --output outputs/test
```

## 评估

测试集按原始顺序评估；每类的第一个样本由标签索引自动确定。噪声在分批推理之前统一生成，因此不随推理批大小改变。训练、验证和测试的随机种子互相独立。

```powershell
python -m denoise.evaluate --checkpoint runs/full/model.pt --ensemble --project-output --external examples/image.png --output outputs/example
python -m denoise.audit outputs/example
```

`--external` 接受自行提供的图片并转换为灰度，不缩放尺寸。外部图片不参与训练或模型选择。输出包括逐图 CSV、汇总 JSON、保存浮点像素的 NPZ 和对比图。`--project-output` 仅对最终重建结果施加灰度范围约束；原始输出指标仍完整保留。图像显示范围固定为 `[0, 1]`，不会改写计算指标的数组。

默认基线为加噪输入及固定参数的 `3 × 3` 高斯平滑。自集成需要八次前向计算，是否启用应依据验证集决定。MNIST 是手写数字分布，在自然图像上的效果需要单独检验。

## CPU 运行

`--device auto` 自动选择可用计算后端。`--device cpu` 可强制使用 CPU，无需修改程序；CPU 使用完整精度，关闭混合精度与梯度缩放。CPU 配置采用较小的批大小，网络结构保持一致。

```powershell
python -m denoise.train --config configs/cpu.json --device cpu --output runs/cpu
python -m denoise.evaluate --checkpoint runs/full/model.pt --device cpu --batch-size 16 --ensemble --project-output --output outputs/cpu
```

训练配置、随机噪声和优化更新会影响结果。CPU 配置是独立的训练方案，不能将其未经完整训练的输出视为其他配置的复现结果。

## 结构

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

数据集、权重、运行记录和生成结果由 `.gitignore` 排除。仅加载可信来源的训练断点；完整续训状态包含优化器对象。固定种子支持实验复现，但不同设备或 PyTorch 后端不保证逐位一致。

## 参考文献

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Liu et al. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

本项目自行实现各网络组件，仅采用公开的架构思想。
