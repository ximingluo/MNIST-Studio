import os
import sys
import torch
from tqdm import tqdm

# 将项目 src 目录加入 python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

# 导入自定义模块
from data.dataset import get_data_loader
from data.sampler import create_mistake_sampler
from model import res18
from utils.checkpoint import load_model, get_model_path
from utils.eval import evaluate
from utils.visualize import visualize_incorrect_predictions

def main():
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 实例化并加载模型
    model = res18(in_channels=3, num_classes=10)
    model.to(device)
    
    # 加载已保存的模型权重
    model_path = get_model_path("resnet18_mnist")
    model, metadata = load_model(model, model_path, device=device)
    
    if model is None:
        print("Error: Could not load model. Please run training script first.")
        return
        
    if metadata:
        print(f"Loaded model from Epoch {metadata.get('epoch', 'unknown')} with accuracy {metadata.get('accuracy', 0):.4f}")

    # 3. 获取测试数据加载器
    test_data = get_data_loader(is_train=False, batch_size=128)
    
    # 4. 初始化错题集采样器
    sampler = create_mistake_sampler()

    # 5. 执行全面评估并收集错题
    print("\nEvaluating and collecting misclassified samples...")
    model.eval()
    n_correct = 0
    n_total = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_data)):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            
            # 统计准确率
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
            
            # 自动保存错题到 data/mistakes
            sampler.collect_batch(x, y, preds, batch_idx * test_data.batch_size)
    
    print(f"Final Test Accuracy: {n_correct / n_total:.4f}")
    
    # 6. 可视化分析部分错误样本
    print("\nVisualizing 10 misclassified images for error analysis...")
    visualize_incorrect_predictions(model, test_data, device, num_images=10, delay=0)

if __name__ == "__main__":
    main()
