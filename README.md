# Transformer Denoising

**简体中文** · [English](README.en.md)

使用 PyTorch 从零训练的灰度图像去噪模型。网络采用重叠图像块、窗口自注意力、二维相对位置偏置和 SwiGLU，不含卷积层，也不依赖现成去噪模型或外部预训练权重。

## 模型与数据

- 原始 MNIST 像素仅除以 255。GT 始终为原始干净图像，不旋转、翻转、裁剪或改变灰度。
- 独立添加均值为 0、标准差为 50/255 的高斯噪声，不裁剪含噪输入。
- 网络内部计算相对灰度特征，通过输入残差保留绝对亮度；特征计算不修改训练目标。
- 8 个窗口 Transformer 块、64 维特征、4 个注意力头，共 338,660 个参数。
- 同一模型直接处理 28×28、32×32 及其他受支持尺寸，输出保持输入尺寸。
- 仅使用平均 L1 训练；最终重建投影至 [0,1]，原始输出指标同时保留。

## 训练

需要 Python 3.11 或更新版本。安装适合计算后端的 PyTorch，再安装依赖。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m denoise.train --config configs/train.json --output runs/original
```

全部 60,000 张原始训练图像按固定轮数训练，官方测试集不参与训练。数据下载至 `data/`。`model.pt` 保存 EMA 权重，`last.pt` 保存完整恢复状态；最终采用固定末轮检查点。

```powershell
python -m denoise.train --config configs/train.json --output runs/original --resume runs/original/last.pt
```

恢复要求配置、原始数据摘要及目标协议一致。程序拒绝不兼容旧检查点，不静默覆盖已有训练。仅加载可信来源的完整训练断点。

## 评估与核验

```powershell
python -m denoise.evaluate --checkpoint runs/original/model.pt --split train-first --external examples/image.png --output outputs/required
python -m denoise.evaluate --checkpoint runs/original/model.pt --split test --seed 20261029 --output outputs/test
python -m denoise.audit outputs/required --external examples/image.png
python -m denoise.audit outputs/test
```

`examples/image.png` 表示自行提供的灰度图片。图片保持原尺寸，只对副本加噪。`train-first` 按原始训练集顺序选取每类首样本；`test` 评估官方 10,000 张测试图像。两类结果明确区分。

噪声在分批推理前生成，不随批大小变化。输出包括三联对比图、逐图 CSV、汇总 JSON 和浮点 NPZ。PSNR 固定采用峰值 1；同时报告逐图平均 PSNR 和总体 MSE 对应的 PSNR。绘图的 [0,1] 显示范围不改写含噪评价数据。

## CPU 备选

`--device auto` 自动选择可用后端；`--device cpu` 强制 CPU。CPU 支持完整训练、同环境精确断点恢复和同一检查点推理，不启用混合精度。

```powershell
python -m denoise.train --config configs/cpu.json --device cpu --output runs/cpu
python -m denoise.evaluate --checkpoint runs/original/model.pt --device cpu --batch-size 16 --split test --seed 20261029 --output outputs/cpu
```

CPU 配置采用较小批大小。沿用主实验超参数时，可对 `configs/train.json` 指定 `--device cpu`。不同计算后端的浮点结果可能存在微小差异。

## 结构

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

数据、权重、运行记录和生成结果由 `.gitignore` 排除。自然图像与 MNIST 存在分布差异，单幅图片的重建结果不代表一般自然图像去噪性能。
