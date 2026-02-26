"""
错题集采样模块 - 负责收集并保存模型预测错误的样本，以及提供增强的数据加载功能
"""
import os
import re
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image

class MistakeSampler:
    """
    错题集采样器：用于捕获验证/测试过程中的 Bad Cases
    """
    def __init__(self, save_dir="src/data/mistakes"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # MNIST 归一化参数 (与 dataset.py 保持一致)
        self.mean = torch.tensor([0.1307, 0.1307, 0.1307]).view(3, 1, 1)
        self.std = torch.tensor([0.3081, 0.3081, 0.3081]).view(3, 1, 1)

    def save_mistake(self, image_tensor, target, pred, index):
        """
        保存单个错误样本为图片文件
        文件名格式: idx{索引}_t{真实标签}_p{预测标签}.png
        """
        # 1. 反归一化还原图像
        img = image_tensor.cpu() * self.std + self.mean
        img = img.clamp(0, 1)
        
        # 2. 转换为 PIL 对象并保存
        img_pil = transforms.ToPILImage()(img)
        filename = f"idx{index}_t{target}_p{pred}.png"
        img_pil.save(os.path.join(self.save_dir, filename))

    def collect_batch(self, images, targets, preds, batch_start_idx):
        """
        从一个批次中自动筛选并收集所有错误样本
        """
        # 找出预测错误的掩码
        mask = (preds != targets)
        incorrect_indices = mask.nonzero(as_tuple=True)[0]
        
        for i in incorrect_indices:
            self.save_mistake(
                images[i], 
                targets[i].item(), 
                preds[i].item(), 
                batch_start_idx + i.item()
            )
        return len(incorrect_indices)

def create_mistake_sampler(save_dir="src/data/mistakes"):
    """创建采样器实例的便捷函数"""
    return MistakeSampler(save_dir=save_dir)


class MistakeDataset(Dataset):
    """
    错题集数据集：加载 src/data/mistakes 目录下的 PNG 图像
    文件名解析规则: idx{index}_t{target}_p{pred}.png
    """
    def __init__(self, mistakes_dir="src/data/mistakes", transform=None):
        self.mistakes_dir = mistakes_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []
        
        if os.path.exists(mistakes_dir):
            for filename in os.listdir(mistakes_dir):
                if filename.endswith(".png"):
                    # 使用正则提取真实标签 _t{target}_
                    match = re.search(r'_t(\d+)_', filename)
                    if match:
                        self.image_paths.append(os.path.join(mistakes_dir, filename))
                        self.targets.append(int(match.group(1)))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target = self.targets[idx]
        
        # 加载并转换图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, target


def get_data_loader_with_mistakes(batch_size=128, mistakes_dir="src/data/mistakes", repeat_count=1):
    """
    获取数据加载器，支持选择性地合并错题集
    
    Args:
        batch_size: 批量大小
        mistakes_dir: 错题集路径
        repeat_count: 错题集数据重复次数，默认为 1 (不重复)
    """
    # 定义数据转换：将图像转换为 Tensor
    # 保持与 dataset.py 一致的预处理逻辑
    to_tensor = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)) # 3通道归一化
    ])
    
    data_set = MistakeDataset(mistakes_dir=mistakes_dir, transform=to_tensor)
    
    # 如果需要增加错题集的权重，可以将数据集重复多次
    if repeat_count > 1:
        data_set = ConcatDataset([data_set] * repeat_count)
        
    # 返回数据加载器
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)
