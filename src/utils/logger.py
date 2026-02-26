"""
训练日志工具模块
提供 TensorBoard 集成、训练日志记录等功能
"""

import os
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    """
    训练日志记录器
    封装 TensorBoard SummaryWriter，提供便捷的日志记录接口
    """
    
    def __init__(self, log_dir, experiment_name="default"):
        """
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.epoch = 0
        
    def log_training_loss(self, loss, step=None):
        """记录训练损失"""
        if step is None:
            step = self.global_step
        self.writer.add_scalar('training loss', loss, step)
        
    def log_accuracy(self, accuracy, phase="test", epoch=None):
        """记录准确率"""
        if epoch is None:
            epoch = self.epoch
        self.writer.add_scalar(f'{phase} accuracy', accuracy, epoch)
        
    def log_learning_rate(self, lr):
        """记录学习率"""
        self.writer.add_scalar('learning rate', lr, self.epoch)
        
    def log_feature_maps(self, feature_map, layer_name, step=None):
        """记录特征图"""
        if step is None:
            step = self.global_step
        self.writer.add_image(f'features/{layer_name}', feature_map, step)
        
    def log_scalar(self, name, value, step=None):
        """记录任意标量"""
        if step is None:
            step = self.global_step
        self.writer.add_scalar(name, value, step)
        
    def step(self):
        """全局步数 +1"""
        self.global_step += 1
        
    def next_epoch(self):
        """进入下一个 epoch"""
        self.epoch += 1
        
    def close(self):
        """关闭 writer"""
        self.writer.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_logger(experiment_name, log_dir="logs"):
    """
    创建训练日志记录器
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志根目录
    
    Returns:
        TrainLogger: 日志记录器
    """
    return TrainLogger(log_dir=log_dir, experiment_name=experiment_name)
