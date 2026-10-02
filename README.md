# U-Net PennFudanPed 语义分割实践

这是我第一次完整复现语义分割训练流程的学习项目。目前版本以“完整跑通”为主要目标：将原 `Pytorch-UNet` 项目面向 Carvana 汽车分割的数据流程，适配为基于 `PennFudanPed` 数据集的行人二分类语义分割。

> 当前状态：`v0.1` 初版。训练、验证、checkpoint 保存和自训练模型预测均已跑通，后续会继续优化分割效果和训练策略。

## 项目目标

在已有图像分类经验的基础上，完整理解并实践下面这条语义分割链路：

```text
PennFudanPed 原始数据
        ↓
实例 mask 转二值 mask
        ↓
Dataset / DataLoader
        ↓
U-Net forward
        ↓
CrossEntropy + Dice Loss
        ↓
Backward / Optimizer
        ↓
Validation Dice
        ↓
Checkpoint
        ↓
使用自己训练的模型预测
```

本项目只包含两个语义类别：

```text
0：background
1：person
```

模型配置为：

```text
n_channels = 3
n_classes = 2
```

## 主要改动

本项目基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 修改，保留了原项目的 U-Net 结构、Dataset 设计、Dice 实现和核心训练逻辑。主要改动如下：

1. 使用小型 `PennFudanPed` 行人数据集替代体积较大的 Carvana 数据集。
2. 新增 `prepare_pennfudan.py`：
   - 将 PennFudan 的 instance mask 合并为 person 二值 mask；
   - 输出 mask 像素值为 `0/255`；
   - 将不同尺寸的图片和 mask 等比例缩放并 padding 到 `512×512`；
   - 保留原始数据，不在原目录中修改文件。
3. 新增 `check_data.py`，在训练前检查图片与 mask 匹配、shape、dtype、类别值和 DataLoader batch。
4. 新增 `validate.py`，可以独立验证指定 checkpoint。
5. 验证 DataLoader 使用 `drop_last=False`，避免小型验证集丢失最后一个 batch。
6. W&B 改为可选依赖：未安装时仍可正常完成本地训练。
7. 训练日志增加每个 epoch 的平均 loss。

## 项目结构

```text
U-Net-PennFudanPed/
├── unet/
│   ├── __init__.py
│   ├── unet_model.py
│   └── unet_parts.py
├── utils/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── dice_score.py
│   └── utils.py
├── prepare_pennfudan.py
├── check_data.py
├── train.py
├── evaluate.py
├── validate.py
├── predict.py
├── requirements.txt
├── LICENSE
└── README.md
```

数据集、训练权重和预测输出不会上传到仓库。

## 环境

本次实际运行环境：

```text
Windows
PyTorch 2.7.0+cu118
TorchVision 0.22.0+cu118
CUDA 11.8
NVIDIA GeForce RTX 3060 Laptop GPU
```

建议先根据自己的显卡环境，从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配的 PyTorch 和 TorchVision，再安装其余依赖：

```powershell
pip install -r requirements.txt
```

## 1. 下载 PennFudanPed

数据集来自 PyTorch 官方教程使用的 Penn-Fudan 行人检测与分割数据集：

```powershell
Invoke-WebRequest `
  -Uri "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip" `
  -OutFile ".\PennFudanPed.zip"

Expand-Archive -LiteralPath ".\PennFudanPed.zip" -DestinationPath "."
```

解压后的原始结构应为：

```text
PennFudanPed/
├── PNGImages/
└── PedMasks/
```

## 2. 准备语义分割数据

```powershell
python .\prepare_pennfudan.py `
  --source .\PennFudanPed `
  --output .\data `
  --size 512
```

处理后结构：

```text
data/
├── imgs/
│   ├── FudanPed00001.png
│   └── ...
└── masks/
    ├── FudanPed00001_mask.png
    └── ...
```

若需要重新生成已有文件，增加 `--overwrite`。

## 3. 训练前检查数据

```powershell
python .\check_data.py --scale 0.5 --batch-size 2 --samples 3
```

预期关键输出：

```text
Images: 170
Masks: 170
Images without masks: []
Masks without images: []
image shape: torch.Size([3, 256, 256])
mask shape: torch.Size([256, 256])
mask unique: tensor([0, 1])
batch image shape: torch.Size([2, 3, 256, 256])
batch mask shape: torch.Size([2, 256, 256])
```

## 4. 训练

本次初版使用 5 个 epoch，只验证完整流程：

```powershell
python .\train.py -e 5 -b 2 -s 0.5 --classes 2 --amp
```

如果没有 CUDA，请去掉 `--amp`：

```powershell
python .\train.py -e 5 -b 2 -s 0.5 --classes 2
```

训练集和验证集按 `90%/10%` 划分，并使用固定随机种子 `0`。每个 epoch 结束后会生成：

```text
checkpoints/checkpoint_epoch1.pth
...
checkpoints/checkpoint_epoch5.pth
```

## 5. 独立验证 checkpoint

```powershell
python .\validate.py `
  --model .\checkpoints\checkpoint_epoch5.pth `
  --batch-size 2 `
  --scale 0.5 `
  --classes 2 `
  --amp
```

## 6. 使用自己训练的模型预测

将一张行人图片放到 `test_images`，然后执行：

```powershell
New-Item -ItemType Directory -Force .\outputs

python .\predict.py `
  --model .\checkpoints\checkpoint_epoch5.pth `
  --input .\test_images\person.png `
  --output .\outputs\person_mask.png `
  --scale 0.5 `
  --classes 2
```

如果本地 Matplotlib/OpenMP 环境正常，可以增加 `--viz` 显示结果。

## 初版实验结果

| 配置 | 数值 |
|---|---:|
| 原始图片 / mask | 170 / 170 |
| 训练集 / 验证集 | 153 / 17 |
| Epochs | 5 |
| Batch size | 2 |
| Scale | 0.5 |
| Optimizer | RMSprop |
| Loss | CrossEntropy + Dice Loss |
| 最终 checkpoint 验证 Dice | 0.701290 |
| 单张验证图片 Dice | 0.661027 |

这些结果只代表小数据集、少量 epoch、从零训练情况下的初步实验，不能与 Carvana 官方成绩或其他正式基准直接比较。目前预测仍可能将树木、阴影等背景误判为人物。

## 后续优化计划

- 增加同步的 image/mask 数据增强；
- 增加训练 epoch，并保存最佳验证 checkpoint；
- 对比不同学习率、优化器和 scheduler；
- 增加 IoU、Precision、Recall 等指标；
- 改进类别不平衡处理；
- 尝试预训练编码器或其他 U-Net 变体；
- 增加预测叠加图和训练曲线；
- 增加测试和更完整的实验记录。

## 参考与许可

- U-Net 论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- 原始代码：[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- PennFudanPed 教程：[TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

本项目是对 GPLv3 项目的修改与学习实践，继续遵循仓库中的 [GNU GPL v3](LICENSE)。
