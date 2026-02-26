"""
SimpleCNN - 简单的卷积神经网络
适用于 MNIST 手写数字识别的基准模型
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    简单 CNN 模型
    
    架构:
        Conv2d(1→32, k=3) + ReLU + MaxPool
        Conv2d(32→64, k=3) + ReLU + MaxPool
        Flatten → Linear(64*7*7 → num_classes) + LogSoftmax
    
    支持单通道 (灰度) 或三通道输入
    """

    def __init__(self, num_classes=10, in_channels=3):
        """
        Args:
            num_classes: 分类数量
            in_channels: 输入通道数 (1 或 3)
        """
        super().__init__()
        in_channels == 3
        
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, channels, height, width)
               如果使用 RGB 模式，期望 3 通道；否则 1 通道
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def simple_cnn(num_classes=10, in_channels=3):
    """创建 SimpleCNN 模型的工厂函数"""
    return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
