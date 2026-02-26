"""
评估工具模块
提供模型评估、准确率计算等功能
"""

import torch


def evaluate(model, data_loader, device):
    """
    评估模型在数据集上的准确率
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        float: 准确率 (0-1 之间)
    """
    model.eval()
    n_correct = 0
    n_total = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    
    return n_correct / n_total


def evaluate_with_loss(model, data_loader, device, loss_fn):
    """
    评估模型并返回准确率和平均损失
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        device: 计算设备
        loss_fn: 损失函数
    
    Returns:
        tuple: (准确率，平均损失)
    """
    model.eval()
    n_correct = 0
    n_total = 0
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    
    accuracy = n_correct / n_total
    avg_loss = total_loss / n_batches
    
    return accuracy, avg_loss
