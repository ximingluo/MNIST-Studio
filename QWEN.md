# MNIST 手写数字识别研究平台

基于 PyTorch 的模块化深度学习框架，专注于手写数字识别任务的模型对比研究。

## 目录结构

```
CNN/
├── scripts/                # 训练和评估脚本
│   ├── train.py            # 训练脚本入口
│   └── eval.py             # 评估脚本入口
│
├── src/                    # 源代码目录
│   ├── model/              # 模型定义
│   │   ├── __init__.py     # 模型注册表
│   │   ├── mlp.py          # 多层感知机
│   │   ├── resnet.py       # ResNet 风格网络
│   │   └── simple_cnn.py   # 简单 CNN 基准
│   │
│   ├── data/               # 数据加载
│   │   ├── __init__.py
│   │   ├── dataset.py      # MNIST 数据加载器
│   │   └── sampler.py      # 错题集采样器
│   │
│   └── utils/              # 工具模块
│       ├── __init__.py
│       ├── checkpoint.py   # 模型保存/加载
│       ├── eval.py         # 评估函数
│       ├── logger.py       # TensorBoard 日志
│       └── visualize.py    # 可视化函数
│
├── outputs/                # 输出目录
│   └── models/             # 保存训练好的模型权重
│
├── logs/                   # TensorBoard 日志目录
│
├── tests/                  # 测试目录
│
├── .gitignore              # Git 忽略配置
└── QWEN.md                 # 项目文档
```

## 快速开始

### 1. 训练模型

```bash
python scripts/train.py
```

### 2. 评估模型

```bash
python scripts/eval.py
```

### 3. 查看训练日志

```bash
tensorboard --logdir=logs
```

然后在浏览器打开 http://localhost:6006

## 可用模型

| 模型名称 | 描述 | 输入通道 | 主要特点 |
|---------|------|---------|---------|
| `mlp` | 多层感知机 | 3 (展平) | 2 层隐藏层，256→128 |
| `simple_cnn` | 简单 CNN 基准 | 1 或 3 | 2 层卷积 + MaxPool |
| `res18` | ResNet-18 风格 | 1 或 3 | BasicBlock 残差结构 |

### 模型架构详情

**MLP (`src/model/mlp.py`)**
```
Linear(in_features → 256) + ReLU
Linear(256 → 128) + ReLU
Linear(128 → num_classes) + LogSoftmax
```

**SimpleCNN (`src/model/simple_cnn.py`)**
```
Conv2d(in→32, k=3) + ReLU + MaxPool(2,2)
Conv2d(32→64, k=3) + ReLU + MaxPool(2,2)
Flatten → Linear(64*7*7 → num_classes) + LogSoftmax
```

**ResNet-18 (`src/model/resnet.py`)**
```
Conv1: Conv2d + BN + ReLU
Layer1: BasicBlock × 2 (64 通道)
Layer2: BasicBlock × 2 (128 通道, stride=2)
Layer3: BasicBlock × 2 (256 通道, stride=2)
Layer4: BasicBlock × 2 (512 通道, stride=2)
Head: AdaptiveAvgPool → FC → num_actions
```

## 核心模块

### 数据模块 (`src/data/`)

- **`get_data_loader()`**: 获取 MNIST 数据加载器
  - 自动下载数据集到 `src/data/database/`
  - 预处理：Grayscale→RGB → ToTensor → Normalize
  - 归一化参数：mean=0.1307, std=0.3081

- **`MistakeSampler`**: 错题集采样器
  - 自动收集预测错误的样本
  - 保存到 `src/data/mistakes/` 目录
  - 文件名格式：`idx{索引}_t{真实标签}_p{预测标签}.png`

- **`get_data_loader_with_mistakes()`**: 支持合并错题集的数据加载器
  - `repeat_count`: 错题集数据重复次数

### 工具模块 (`src/utils/`)

| 模块 | 功能 |
|-----|------|
| `checkpoint.py` | 模型保存/加载，检查点管理 |
| `eval.py` | 模型评估，准确率/损失计算 |
| `logger.py` | TensorBoard 集成，训练日志记录 |
| `visualize.py` | 预测可视化，特征图可视化，错题分析 |

## 训练特性

- **梯度裁剪**: 防止梯度爆炸 (`max_norm=1.0`)
- **TensorBoard 集成**: 实时可视化训练过程
- **特征图可视化**: 记录中间层激活
- **错题分析**: 自动收集并可视化错误样本
- **模型检查点**: 自动保存模型权重和元数据

## 输出说明

### 模型保存位置

- 默认路径：`outputs/models/<model_name>.pth`
- 格式：包含 `model_state_dict` 和可选的 `metadata`

### 日志位置

- TensorBoard 日志：`logs/<experiment_name>/<timestamp>/`
- 记录内容：Training Loss、Test Accuracy、Feature Maps

### 错题集位置

- 路径：`src/data/mistakes/`
- 格式：PNG 图像，文件名包含真实标签和预测标签

## 扩展指南

### 添加新模型

1. 在 `src/model/` 目录创建新文件，如 `my_model.py`
2. 实现模型类 (继承 `nn.Module`)
3. 在 `src/model/__init__.py` 中导入并导出

```python
# src/model/__init__.py
from .my_model import MyModel
```

### 自定义训练流程

修改 `scripts/train.py` 中的训练逻辑:

```python
# 修改模型实例化
model = my_model(in_features=xxx, num_classes=10)

# 修改优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 修改训练轮数
for epoch in range(10):
    ...
```

### 使用错题集增强训练

```python
from src.data import get_data_loader_with_mistakes

# 使用错题集进行训练
train_data = get_data_loader_with_mistakes(batch_size=128, repeat_count=10)
```

## 常见问题

### 模型文件未找到

运行评估脚本前，先运行训练脚本生成模型权重:

```bash
python scripts/train.py  # 先生成模型
python scripts/eval.py   # 再评估
```

### CUDA 内存不足

减小 `batch_size` 参数:

```python
train_data = get_data_loader(batch_size=64, is_train=True)  # 默认 128
```

### 加载预训练模型

```python
from src.utils.checkpoint import load_model, get_model_path

model_path = get_model_path("mlp_mnist")
model, metadata = load_model(model, model_path, device=device)
```

## 项目约定

- **代码风格**: 使用类型注解和文档字符串
- **日志记录**: 所有实验通过 TensorBoard 记录
- **模型命名**: `<model_name>_<dataset>` 格式 (如 `mlp_mnist`)
- **数据预处理**: 统一使用 MNIST 归一化参数
