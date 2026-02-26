"""
模型检查点工具模块
提供模型保存、加载等功能
"""

import os
import torch


def save_model(model, path, metadata=None):
    """
    保存模型权重
    
    Args:
        model: 神经网络模型
        path: 保存路径
        metadata: 额外元数据 (如 epoch、accuracy 等)
    
    Returns:
        str: 实际保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")
    return path


def load_model(model, path, device=None):
    """
    加载模型权重
    
    Args:
        model: 神经网络模型
        path: 权重文件路径
        device: 计算设备 (默认自动检测)
    
    Returns:
        tuple: (model, metadata) 加载成功返回模型和元数据，失败返回 (None, None)
    """
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None, None
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    
    metadata = checkpoint.get('metadata', None)
    return model, metadata


def get_model_path(model_name, models_dir="outputs/models"):
    """
    根据模型名称生成标准保存路径
    
    Args:
        model_name: 模型名称
        models_dir: 模型保存根目录
    
    Returns:
        str: 模型保存路径
    """
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"{model_name}.pth")