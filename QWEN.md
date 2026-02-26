# MNIST 手写数字识别研究平台

基于 PyTorch 的模块化深度学习框架，专注于手写数字识别任务的模型对比研究。

## 目录结构

```
CNN/
├── run.py                  # 主入口：训练、对比、TensorBoard
├── requirements.txt        # Python 依赖
├── .gitignore              # Git 忽略配置
├── QWEN.md                 # 项目文档
│
├── models/                 # 保存训练好的模型权重
│   └── *.pth
│
├── logs/                   # TensorBoard 日志目录
│   └── <experiment_name>/
│
└── src/                    # 源代码目录
    ├── train.py            # 训练脚本 (命令行接口)
    ├── trainer.py          # 训练器封装
    │
    ├── model/              # 模型定义
    │   ├── __init__.py     # 模型注册表
    │   ├── simple_cnn.py   # 简单 CNN 基准
    │   ├── efficientnet.py # EfficientNet 系列
    │   └── resnet.py       # ResNet 风格网络
    │
    ├── data/               # 数据加载
    │   ├── __init__.py
    │   └── mnist.py        # MNIST 数据加载器
    │
    └── utils/              # 工具模块
        ├── __init__.py
        ├── eval.py         # 评估函数
        ├── visualize.py    # 可视化函数
        ├── checkpoint.py   # 模型保存/加载
        └── logger.py       # TensorBoard 日志
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认配置训练 EfficientNet-B0
python src/train.py --model efficientnet_b0

# 指定实验名称和训练参数
python src/train.py --model simple_cnn --experiment baseline --epochs 20 --lr 0.001

# 查看所有可用模型
python src/train.py --list_models
```

### 3. 使用主入口脚本

```bash
# 训练单个模型
python run.py train --model efficientnet_b0 --epochs 20

# 批量对比多个模型
python run.py compare --epochs 10

# 启动 TensorBoard
python run.py tensorboard
```

### 4. 查看训练日志

```bash
tensorboard --logdir=logs
```

然后在浏览器打开 http://localhost:6006

## 可用模型

| 模型名称 | 描述 | 输入通道 | 参数量 |
|---------|------|---------|--------|
| `simple_cnn` | 简单 CNN 基准 (2 层卷积) | 1 (灰度) | ~45K |
| `simple_cnn_gray` | 简单 CNN (显式灰度) | 1 (灰度) | ~45K |
| `efficientnet_b0` | EfficientNet-B0 | 3 (RGB) | ~4M |
| `efficientnet_v2_s` | EfficientNetV2-S | 3 (RGB) | ~20M |
| `dqn_res18` | ResNet-18 DQN | 1 (灰度) | ~11M |
| `ddqn_res18` | ResNet-18 Dueling DQN | 1 (灰度) | ~11M |

## 训练参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--model` | efficientnet_b0 | 模型名称 |
| `--experiment` | default | 实验名称 (用于日志) |
| `--epochs` | 10 | 训练 epoch 数 |
| `--batch_size` | 64 | 批量大小 |
| `--lr` | 0.0001 | 初始学习率 |
| `--image_size` | 48 | 输入图像尺寸 |
| `--target_accuracy` | 0.99 | 目标准确率 (提前停止) |
| `--resume` | None | 从检查点恢复 |

## 模型对比研究

### 示例：对比不同架构

```bash
# 对比 CNN、EfficientNet-B0、EfficientNet-V2-S
python run.py compare --models simple_cnn efficientnet_b0 efficientnet_v2_s --epochs 15
```

### 示例：学习率消融实验

```bash
# 不同学习率对 simple_cnn 的影响
python src/train.py --model simple_cnn --experiment lr_0.01 --lr 0.01
python src/train.py --model simple_cnn --experiment lr_0.001 --lr 0.001
python src/train.py --model simple_cnn --experiment lr_0.0001 --lr 0.0001
```

## 扩展指南

### 添加新模型

1. 在 `src/model/` 目录创建新文件，如 `my_model.py`
2. 实现模型类 (继承 `nn.Module`)
3. 在 `src/model/__init__.py` 中注册:

```python
from .my_model import MyModel

MODEL_REGISTRY = {
    # ... 现有模型
    "my_model": lambda num_classes=10: MyModel(num_classes=num_classes),
}
```

### 添加新数据集

1. 在 `src/data/` 目录创建新文件
2. 实现 `get_data_loader()` 接口
3. 在训练脚本中导入使用

### 自定义训练流程

继承 `src/trainer.py` 中的 `Trainer` 类，重写需要的方法:

```python
from trainer import Trainer

class MyTrainer(Trainer):
    def train_epoch(self, train_loader):
        # 自定义训练逻辑
        pass
```

## 训练特性

- **自动学习率调整**: ReduceLROnPlateau 调度器
- **梯度裁剪**: 防止梯度爆炸 (max_norm=1.0)
- **早停机制**: 达到目标准确率自动停止
- **模型检查点**: 自动保存最佳模型
- **TensorBoard 集成**: 实时可视化训练过程
- **特征图可视化**: 记录中间层激活

## 数据预处理

默认预处理流程:
1. Resize 到指定尺寸 (默认 48x48)
2. Grayscale → RGB (仅 EfficientNet 系列)
3. ToTensor
4. Normalize (MNIST 统计值)

## 输出说明

### 模型保存位置

- 训练过程：`models/<model_name>_<experiment>.pth`
- 格式：包含 `model_state_dict` 和 `metadata`

### 日志位置

- TensorBoard 日志：`logs/<model_name>_<experiment>/`
- 记录内容：Loss、Accuracy、Learning Rate、Feature Maps

## 常见问题

### CUDA 内存不足

减小 `--batch_size` 或 `--image_size`:

```bash
python src/train.py --model efficientnet_b0 --batch_size 32 --image_size 32
```

### 加载预训练模型

```python
from model import get_model
from utils.checkpoint import load_model

model = get_model('efficientnet_b0')
model, metadata = load_model(model, 'models/my_model.pth')
```

### 多 GPU 训练

当前版本支持单 GPU，多 GPU 需修改 `trainer.py` 使用 `DataParallel`:

```python
model = nn.DataParallel(model)
```
