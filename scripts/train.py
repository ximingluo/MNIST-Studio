import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

import torch
import torchvision
from tqdm import tqdm

# 导入自定义模块
from src.data import get_data_loader, get_data_loader_with_mistakes
from src.model import res18, simple_cnn, mlp
from src.utils.checkpoint import save_model, load_model, get_model_path
from src.utils.eval import evaluate
from src.utils.logger import create_logger
from src.utils.visualize import FeatureMapHook, visualize_predictions, visualize_feature_maps

def main():
    # 获取数据加载器
    # train_data = get_data_loader_with_mistakes(batch_size=128, repeat_count=100)
    # test_data = get_data_loader_with_mistakes(batch_size=128, repeat_count=1)
    train_data = get_data_loader(batch_size=128, is_train=True)
    test_data = get_data_loader(batch_size=512, is_train=False)
    
    # 实例化模型
    model = mlp(in_features=3*28*28, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # 加载已有模型
    model_path = get_model_path("mlp_mnist")
    load_model(model, model_path)

    # 初始化日志和 Hook
    logger = create_logger("mlp_experiment")
    hook = FeatureMapHook()
    
    # 注册特征图捕获 Hook (SimpleCNN 特征层)
    # hook.register_hook(model.features[0], 'conv1')
    # hook.register_hook(model.features[3], 'conv2')

    print(f"Initial accuracy: {evaluate(model, test_data, device):.4f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
    # 训练循环
    for epoch in range(3):
        model.train()
        loop = tqdm(train_data, desc=f'Epoch {epoch + 1}')
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新进度条和日志
            loop.set_postfix(loss=loss.item())
            if logger.global_step % 500 == 0:
                logger.log_training_loss(loss.item())
                # 记录特征图
                for name, act in hook.activations.items():
                    if act is not None:
                        grid = visualize_feature_maps(act, name)
                        logger.log_feature_maps(grid, name)
            
            logger.step()
        
        # Epoch 结束处理
        acc = evaluate(model, test_data, device)
        print(f"Epoch {epoch + 1} Accuracy: {acc:.4f}")
        logger.log_accuracy(acc)
        # visualize_predictions(model, test_data, device, num_images=10, delay=0)
        
        if acc > 0.98:
            break
        logger.next_epoch()

    logger.close()
    save_model(model, model_path)
    visualize_predictions(model, test_data, device, num_images=10, delay=0)

if __name__ == "__main__":
    main()
