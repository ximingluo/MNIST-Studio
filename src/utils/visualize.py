"""
可视化工具模块
提供预测结果可视化、特征图可视化等功能
"""

import torch
import matplotlib.pyplot as plt
import torchvision


def visualize_predictions(model, data_loader, device, num_images=10, delay=0):
    """
    可视化模型预测结果
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        device: 计算设备
        num_images: 展示图片数量
        delay: 每张图展示延时 (秒)，0 表示等待用户手动关闭
    """
    model.eval()
    images_shown = 0
    
    print(f"Starting visualization of {num_images} images...")
    
    with torch.no_grad():
        for x_batch, _ in data_loader:
            for i in range(len(x_batch)):
                if images_shown >= num_images:
                    return
                
                img = x_batch[i]
                img_input = img.unsqueeze(0).to(device)
                
                # 获取预测结果
                logits = model(img_input)
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                predict = torch.argmax(logits).item()
                
                # 创建子图：左图右概率
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # 显示图像
                # 自动处理通道：[C, H, W] -> [H, W, C] 或 [H, W]
                display_img = img.permute(1, 2, 0).cpu().numpy()
                if display_img.shape[2] == 1:
                    display_img = display_img.squeeze(2)
                
                ax1.imshow(display_img, cmap='gray' if len(display_img.shape) == 2 else None)
                ax1.set_title(f"Prediction: {predict}")
                ax1.axis('off')
                
                # 显示概率分布
                ax2.bar(range(10), probs)
                ax2.set_xticks(range(10))
                ax2.set_xlabel('Class')
                ax2.set_ylabel('Probability')
                ax2.set_title("Probabilities")
                ax2.set_ylim(0, 1.1)
                
                # 添加数值标签
                for j, v in enumerate(probs):
                    ax2.text(j, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)
                
                plt.tight_layout()
                images_shown += 1
                
                if delay > 0:
                    plt.show(block=False)
                    plt.pause(delay)
                    plt.close(fig)
                else:
                    plt.show()


def visualize_feature_maps(activations, layer_name, nrow=8):
    """
    可视化特征图
    
    Args:
        activations: 特征图张量 [B, C, H, W]
        layer_name: 层名称
        nrow: 每行显示的通道数
    
    Returns:
        特征图网格 (可用于 TensorBoard)
    """
    # 取批次中第一张图
    feature_map = activations[0]  # [C, H, W]
    # 增加通道维度 [C, 1, H, W]
    feature_map = feature_map.unsqueeze(1)
    # 创建网格
    grid = torchvision.utils.make_grid(feature_map, nrow=nrow, normalize=True, scale_each=True)
    return grid


class FeatureMapHook:
    """
    用于捕获网络中间层特征图的 Hook
    """
    
    def __init__(self):
        self.activations = {}
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hook(self, module, name):
        """为模块注册 hook"""
        module.register_forward_hook(self.get_activation(name))
    
    def get_feature_map(self, name):
        """获取指定层的特征图"""
        return self.activations.get(name)
    
    def clear(self):
        """清空所有激活值"""
        self.activations.clear()
