import torch
import torchvision
from tqdm import tqdm

# 导入自定义模块
from data.dataset import get_data_loader
from model import res18
from utils.checkpoint import save_model, load_model, get_model_path
from utils.eval import evaluate
from utils.logger import create_logger
from utils.visualize import visualize_predictions, FeatureMapHook, visualize_feature_maps

def main():
    # 获取数据加载器
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    
    # 实例化模型
    model = res18(in_channels=3, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # 加载已有模型
    model_path = get_model_path("resnet18_mnist")
    load_model(model, model_path)

    # 初始化日志和 Hook
    logger = create_logger("resnet18_experiment")
    hook = FeatureMapHook()
    
    # 注册特征图捕获 Hook (ResNet-18 特征层)
    hook.register_hook(model.conv1, 'conv1')
    hook.register_hook(model.conv2_x, 'layer1')
    hook.register_hook(model.conv3_x, 'layer2')
    hook.register_hook(model.conv4_x, 'layer3')
    hook.register_hook(model.conv5_x, 'layer4')

    print(f"Initial accuracy: {evaluate(model, test_data, device):.4f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
    # 训练循环
    for epoch in range(1):
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
        print(f"Epoch {epoch} Accuracy: {acc:.4f}")
        logger.log_accuracy(acc)
        save_model(model, model_path, metadata={'accuracy': acc, 'epoch': epoch})
        
        if acc > 0.99:
            break
        logger.next_epoch()

    logger.close()
    visualize_predictions(model, test_data, device, num_images=10, delay=0)

if __name__ == "__main__":
    main()
