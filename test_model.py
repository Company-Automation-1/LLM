#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from model import load_model
from networks import neural_network
from activations import softplus
from util import pt
from mnist.dataset.mnist import load_mnist
from train import MnistTrainer

def main():
    """主函数，加载训练好的模型并进行测试"""
    # 模型文件路径
    model_path = './models/final_model.pkl'
    
    # 创建训练器实例并加载模型
    trainer = MnistTrainer()
    trainer.load_model(model_path)
    
    # 在完整测试集上进行评估
    test_acc = trainer.evaluate(samples=10000)  # 使用全部测试样本
    pt("测试集准确率", f"{test_acc:.4f}")
    
    # 可视化一些预测结果
    visualize_predictions(trainer, num_samples=10)

def visualize_predictions(trainer, num_samples=10):
    """可视化预测结果
    
    参数:
    - trainer: 训练器实例
    - num_samples: 可视化样本数量
    """
    # 加载测试集
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    
    # 随机选择一些样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # 创建图表
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
            
        # 获取样本
        x = x_test[idx]
        true_label = t_test[idx]
        
        # 预测
        output = trainer.predict(x)
        pred_label = np.argmax(output)
        confidence = float(np.max(output))
        
        # 显示图像
        img = x.reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        
        # 设置标题（预测正确显示绿色，错误显示红色）
        title_color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'预测: {pred_label} (实际: {true_label})\n置信度: {confidence:.2f}', 
                         color=title_color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/predictions.png')
    plt.close()
    
    pt("可视化结果已保存", "./results/predictions.png")

if __name__ == "__main__":
    main() 