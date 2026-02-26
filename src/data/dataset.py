import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# 定义一个函数来获取数据加载器
def get_data_loader(is_train):
    # 定义数据转换：将图像转换为 Tensor
    # MNIST 原始尺寸 28x28 对于 EfficientNet 来说太小了
    # Resize 到 48x48 (EfficientNet 最小建议尺寸之一，虽可更小但 48 比较稳)
    # 转为 3 通道 (Grayscale -> RGB)，这样就不需要修改网络的第一层结构，避免初始化问题
    to_tensor = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)) # 3通道归一化
    ])
    # 加载 MNIST 数据集，is_train 决定是训练集还是测试集，应用转换，如果不存在则下载
    data_set = MNIST("src/data/database", is_train, transform=to_tensor, download=True)
    # 返回数据加载器，批量大小为 64，打乱数据
    return DataLoader(data_set, batch_size=64, shuffle=True)
