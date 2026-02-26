# MNIST-Studio

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.8-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 一个模块化的 PyTorch 深度学习框架，专注于 MNIST 手写数字识别的模型对比研究与教学演示

## ✨ 特性

- 🧠 **多模型支持** — 内置 MLP、SimpleCNN、ResNet-18 等多种架构
- 📊 **TensorBoard 可视化** — 实时追踪训练损失、准确率与特征图
- 🎯 **错题分析系统** — 自动收集并可视化错误样本，辅助模型调试
- 🧪 **模块化设计** — 清晰的代码结构，便于扩展新模型与数据集
- 📚 **教学友好** — 详尽的注释与文档，适合深度学习入门学习

## 📦 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- tensorboard
- matplotlib
- tqdm

### 安装依赖

```bash
pip install torch torchvision tensorboard matplotlib tqdm
```

### 训练模型

```bash
# 运行默认训练脚本 (MLP 模型)
python scripts/train.py
```

### 评估模型

```bash
# 评估训练好的模型并可视化错误样本
python scripts/eval.py
```

### 查看训练日志

```bash
# 启动 TensorBoard
tensorboard --logdir=logs

# 在浏览器访问 http://localhost:6006
```

## 🏗️ 项目结构

```
MNIST-Studio/
├── scripts/                 # 可执行脚本
│   ├── train.py             # 训练入口
│   └── eval.py              # 评估入口
│
├── src/                     # 源代码
│   ├── model/               # 模型定义
│   │   ├── __init__.py      # 模型注册表
│   │   ├── mlp.py           # 多层感知机
│   │   ├── resnet.py        # ResNet-18 风格网络
│   │   └── simple_cnn.py    # 简单 CNN 基准
│   │
│   ├── data/                # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py       # MNIST 数据加载器
│   │   └── sampler.py       # 错题集采样器
│   │
│   └── utils/               # 工具函数
│       ├── checkpoint.py    # 模型保存/加载
│       ├── eval.py          # 评估函数
│       ├── logger.py        # TensorBoard 日志
│       └── visualize.py     # 可视化函数
│
├── outputs/models/          # 保存的模型权重
├── logs/                    # TensorBoard 日志
├── src/data/database/       # MNIST 数据集 (自动下载)
└── src/data/mistakes/       # 错题集 (评估时生成)
```

## 🧠 可用模型

| 模型 | 描述 | 参数量 | 典型准确率 |
|------|------|--------|-----------|
| `MLP` | 多层感知机 (256→128) | ~340K | 97%+ |
| `SimpleCNN` | 2 层卷积基准网络 | ~45K | 98%+ |
| `ResNet-18` | 残差网络风格 | ~11M | 99%+ |

### 模型架构对比

```
MLP:          Linear→ReLU→Linear→ReLU→Linear→LogSoftmax
              (展平输入，无空间信息)

SimpleCNN:    Conv→ReLU→Pool → Conv→ReLU→Pool → FC
              (保留空间结构，适合图像)

ResNet-18:    Conv → [BasicBlock×2]×4 → GAP → FC
              (残差连接，更深网络)
```

## 📊 核心功能

### 1. 错题集系统

自动收集模型预测错误的样本，保存到 `src/data/mistakes/` 目录：

```
idx125_t3_p8.png   # 索引 125，真实标签 3，预测为 8
idx892_t7_p1.png   # 索引 892，真实标签 7，预测为 1
```

可用于：
- 分析模型易混淆的数字对
- 针对性增强训练
- 可视化决策边界

### 2. TensorBoard 集成

训练过程中自动记录：
- 训练损失曲线
- 测试准确率
- 卷积层特征图激活

### 3. 可视化分析

```python
# 可视化预测结果
visualize_predictions(model, test_loader, device, num_images=10)

# 可视化错误样本
visualize_incorrect_predictions(model, test_loader, device, num_images=10)

# 可视化特征图
hook = FeatureMapHook()
hook.register_hook(model.features[0], 'conv1')
```

## 🔧 自定义扩展

### 添加新模型

1. 在 `src/model/` 创建新文件 `my_model.py`：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

2. 在 `src/model/__init__.py` 中注册：

```python
from .my_model import MyModel
```

3. 在 `scripts/train.py` 中实例化：

```python
from src.model import MyModel
model = MyModel(num_classes=10)
```

### 使用错题集增强训练

```python
from src.data import get_data_loader_with_mistakes

# 错题集重复 10 次，增强权重
train_loader = get_data_loader_with_mistakes(
    batch_size=128, 
    repeat_count=10
)
```

## 📈 训练技巧

| 技巧 | 效果 | 实现位置 |
|------|------|---------|
| 梯度裁剪 | 防止梯度爆炸 | `clip_grad_norm_(..., max_norm=1.0)` |
| 学习率调整 | 加速收敛 | 可添加 `ReduceLROnPlateau` |
| 错题增强 | 提升难例准确率 | `MistakeSampler` |
| 数据归一化 | 稳定训练 | MNIST 统计量标准化 |

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [PyTorch](https://pytorch.org/) — 深度学习框架
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — 手写数字数据集
- [TensorBoard](https://www.tensorflow.org/tensorboard) — 可视化工具

## 📬 联系方式

如有问题或建议，欢迎通过 Issues 联系！

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

</div>
