# Transformer Denoising

**简体中文** · [English](README.en.md)

基于 PyTorch 的灰度图像去噪实验。模型从零训练，结合重叠图像块、窗口自注意力、二维相对位置偏置与 SwiGLU，支持可变输入尺寸。网络不含卷积层，不调用现成去噪模型或外部预训练权重。

## 方法

- 观测模型：`y = x + ε`，`x ∈ [0, 1]`，`ε ~ N(0, (50/255)²)`；加噪输入不裁剪。
- 重建目标：像素平均 L1；PSNR 固定使用峰值 `1`，分别报告逐图平均值与全体 MSE 对应的 PSNR。
- 网络：步长 `2` 的 `6 × 6` 图像块线性映射，8 个普通/移位窗口 Transformer 块，64 维特征，4 个注意力头，共 338,660 个参数。
- 尺寸适配：反射填充、线性投影与输出裁回；直接支持 `28 × 28`、`32 × 32`，不缩放图像。
- 训练：先学习数字结构，再以亮度、对比度和灰度反转增强拓宽分布；原灰度与变灰度验证 L1 等权用于选择检查点。
- 默认推理：一次前向计算，再投影至 `[0, 1]`。八方向自集成仅作为可选对照。

## 安装与训练

需要 Python 3.11 或更新版本。按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装匹配计算设备的 PyTorch，再安装项目依赖。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denoise.train --config configs/overlap.json --output runs/overlap
python -m denoise.train --config configs/robust.json --output runs/robust --initialize runs/overlap/model.pt
python -m denoise.train --config configs/robust_final.json --output runs/robust_final --full --initialize runs/robust/model.pt
```

MNIST 下载至 `data/`。前两阶段采用固定的 55,000/5,000 训练/验证划分；最后阶段在全部 60,000 张训练图像上按固定预算优化。`model.pt` 保存 EMA 权重，`last.pt` 保存可恢复状态。全训练集模型不能再用于独立验证集选择，程序会拒绝此类评估或初始化。

```powershell
python -m denoise.train --config configs/robust.json --output runs/robust --resume runs/robust/last.pt
python -m denoise.evaluate --checkpoint runs/robust/model.pt --split validation --seed 12345 --ensemble --project-output --output outputs/validation
```

`--initialize` 用于阶段间初始化；`--resume` 恢复同一阶段，要求配置一致。输出目录已有检查点时，新训练不会静默覆盖现有状态。

## 评估与核验

```powershell
python -m denoise.evaluate --checkpoint runs/robust_final/model.pt --project-output --output outputs/test
python -m denoise.audit outputs/test
python -m denoise.evaluate --checkpoint runs/robust_final/model.pt --project-output --external examples/image.png --output outputs/example
```

`examples/image.png` 表示自行提供的图片。外部图片转换为灰度但不缩放；评估程序只读取图片，不更新权重。默认评估官方 10,000 张 MNIST 测试图像，数字样例固定取原始顺序中每类第一个样本。`--split train-first` 单独评估训练集各类首样本，输出明确标记为训练数据。

噪声在分批推理之前统一生成，不依赖推理批大小。输出包括逐图 CSV、汇总 JSON、浮点 NPZ 与对比图。`--project-output` 只约束重建结果，原始输出指标仍被保留。图片采用固定的 `[0, 1]` 显示范围，显示不会修改指标数组。

基线包括含噪输入及固定 `3 × 3` 高斯平滑。`--ensemble` 开启八方向自集成，需八次前向计算。MNIST 与自然纹理存在分布差异，灰度增强不意味着对所有自然图像均有可靠泛化能力。

## 结构增强与重叠重建对照

新增实验保留既有模型与默认入口，比较继续训练、MNIST 结构增强、结构增强加重叠修正分支。结构增强只从当前划分内的 MNIST 干净图像生成仿射裁块与凸组合；随后再加固定强度高斯噪声。混合尺寸训练覆盖 `28 × 28` 和 `32 × 32`。新增重建分支融合浅层与深层特征，以线性映射预测重叠残差块，并按覆盖次数归一化；不使用卷积。

```powershell
python -m denoise.experiment --external assignment/test.png
```

需要先完成前述基础及灰度增强训练。实验按预先固定的三域验证 L1 选择方案，同时约束数字验证 PSNR 的下降不超过 0.20 dB；选定后使用全部 60,000 张图像固定训练 8 轮，最后才读取外部测试图。输出包含三组验证记录、选择依据、新旧模型在三个固定噪声种子下的指标及五列对比图。原始实现和实验驱动的 `denoise` Python 模块总计不超过 800 行，测试检查该约束。

具体配置及数据使用边界见[实验计划](docs/experiment_plan.md)。重新执行相同命令会恢复各阶段训练断点，复用时应保留对应配置和初始化权重。

本轮[实验结果](docs/experiment_results.md)未显示外部测试图有实质改善：验证选定方案的三噪声平均 PSNR 略降，结构增强及重叠修正的事后诊断增益约 0.03 dB，仍低于高斯平滑基线。保留原模型作为默认，新增权重仅作为实验候选。

## CPU 备选

`--device auto` 自动选择可用后端；`--device cpu` 强制使用 CPU。CPU 使用完整精度并关闭梯度缩放，支持训练、断点恢复和同一检查点的推理。

```powershell
python -m denoise.train --config configs/cpu.json --device cpu --output runs/cpu
python -m denoise.evaluate --checkpoint runs/robust_final/model.pt --device cpu --batch-size 16 --project-output --output outputs/cpu
```

CPU 配置采用较小批大小，提供结构相同的独立训练方案。执行完整三阶段方案时，可在对应命令中添加 `--device cpu`。不同配置、设备和后端不保证逐位一致；CPU 小规模验证也不等同于完整实验的重复训练。

## 结构

```text
configs/             Training and comparison configurations
denoise/
  model.py           Linear patch projections and Transformer blocks
  common.py          Data, augmentation, noise, metrics, inference
  train.py           Optimization, selection, and checkpointing
  evaluate.py        Quantitative evaluation and exports
  figures.py         Figures from floating-point arrays
  audit.py           Independent verification of exported metrics
tests/               Structural and numerical contract tests
```

`nonoverlap.json`、`context.json` 和 `final.json` 保留结构及不含灰度增强的对照配置。数据、权重、运行记录与生成结果由 `.gitignore` 排除。仅加载可信来源的完整训练断点。

## 参考文献

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Liu et al. [Swin Transformer](https://arxiv.org/abs/2103.14030).
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
