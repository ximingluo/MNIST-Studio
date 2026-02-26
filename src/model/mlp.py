import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    简单 MLP 模型
    
    架构:
        Linear(in_features → 128) + ReLU
        Linear(128 → num_classes) + LogSoftmax
    
    支持单通道 (灰度) 或三通道输入
    """
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        # 展平输入 (batch_size, channels, height, width) -> (batch_size, channels * height * width)
        x = x.view(x.size(0), -1)
        return self.layers(x)

def mlp(in_features, num_classes):
    """创建基于 MLP 的模型"""
    return MLP(in_features=in_features, num_classes=num_classes)
