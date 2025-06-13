#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

# 导入MNIST数据集
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mnist import load_mnist

# 导入模型相关函数
from activations import softplus
from networks import neural_network

# 定义全局变量
OUTPUT_DIR = './output'

def load_model(model_path):
    """
    加载训练好的模型参数
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 检查是否可用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 将模型参数移动到指定设备上
    weights = [w.to(device) for w in checkpoint['weights']]
    activation_weights = [aw.to(device) for aw in checkpoint['activation_weights']]
    biases = [b.to(device) for b in checkpoint['biases']]
    
    return weights, activation_weights, biases, device

def test_model(model_path, num_samples=10, temperature=1.0):
    """
    测试模型在MNIST测试集上的表现
    
    参数:
    - model_path: 模型参数文件路径
    - num_samples: 随机选择的测试样本数量
    - temperature: softmax温度参数
    """
    # 加载模型参数
    weights, activation_weights, biases, device = load_model(model_path)
    
    # 加载测试数据
    _, (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    
    # 随机选择样本进行测试
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    # 创建图像网格
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # 测试准确率计数
    correct_count = 0
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
            
        # 获取测试样本
        x = torch.tensor(test_images[idx].reshape(784, 1), dtype=torch.float32, device=device)
        true_label = np.argmax(test_labels[idx])
        
        # 前向传播
        output, _, _ = neural_network(x, weights, activation_weights, biases, softplus, temperature)
        
        # 获取预测结果
        predicted_label = torch.argmax(output).item()
        confidence = float(output[predicted_label].cpu().item())
        
        # 检查预测是否正确
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_count += 1
        
        # 显示图像和预测结果
        img = test_images[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        title_color = 'green' if is_correct else 'red'
        axes[i].set_title(f'真实值: {true_label}\n预测值: {predicted_label}\n置信度: {confidence:.2f}', 
                         color=title_color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_samples.png'))
    
    print(f"测试准确率: {correct_count / num_samples:.2f} ({correct_count}/{num_samples})")
    print(f"测试结果已保存至 '{OUTPUT_DIR}/test_samples.png'")
    
    return correct_count / num_samples

def test_custom_image(model_path, image_path, temperature=1.0):
    """
    使用训练好的模型预测自定义图像
    
    参数:
    - model_path: 模型参数文件路径
    - image_path: 自定义图像路径
    - temperature: softmax温度参数
    """
    # 加载模型参数
    weights, activation_weights, biases, device = load_model(model_path)
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('L')  # 转为灰度图
    image = image.resize((28, 28))  # 调整大小为28x28
    
    # 归一化图像
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # 反转颜色（如果需要）
    if img_array.mean() > 0.5:  # 如果图像背景是亮色
        img_array = 1.0 - img_array
    
    # 将图像转换为模型输入格式
    x = torch.tensor(img_array.reshape(784, 1), dtype=torch.float32, device=device)
    
    # 前向传播
    output, _, _ = neural_network(x, weights, activation_weights, biases, softplus, temperature)
    
    # 获取预测结果
    predicted_label = torch.argmax(output).item()
    
    # 获取每个数字的预测概率
    probabilities = output.cpu().reshape(-1).tolist()
    
    # 显示图像和预测结果
    plt.figure(figsize=(8, 6))
    
    # 显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f'预测数字: {predicted_label}')
    plt.axis('off')
    
    # 显示预测概率
    plt.subplot(1, 2, 2)
    barplot = plt.bar(range(10), probabilities)
    plt.xlabel('数字')
    plt.ylabel('概率')
    plt.title('预测概率分布')
    plt.xticks(range(10))
    
    # 高亮显示预测的数字
    barplot[predicted_label].set_color('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'custom_prediction.png'))
    
    print(f"预测结果: {predicted_label}")
    print(f"预测概率分布: {probabilities}")
    print(f"预测概率分布已保存至 '{OUTPUT_DIR}/custom_prediction.png'")
    
    return predicted_label, probabilities

def evaluate_model(model_path, temperature=1.0):
    """
    在整个测试集上评估模型性能
    
    参数:
    - model_path: 模型参数文件路径
    - temperature: softmax温度参数
    """
    # 加载模型参数
    weights, activation_weights, biases, device = load_model(model_path)
    
    # 加载测试数据
    _, (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    
    # 记录分类正确的数量
    correct = 0
    total = len(test_images)
    
    # 每个数字的分类准确率
    digit_correct = [0] * 10
    digit_total = [0] * 10
    
    print(f"正在评估测试集上的性能（共{total}个样本）...")
    
    # 分批处理以避免内存溢出
    batch_size = 100
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch_size_actual = end_idx - i
        
        for j in range(i, end_idx):
            # 获取测试样本
            x = torch.tensor(test_images[j].reshape(784, 1), dtype=torch.float32, device=device)
            true_label = np.argmax(test_labels[j])
            
            # 前向传播
            output, _, _ = neural_network(x, weights, activation_weights, biases, softplus, temperature)
            
            # 获取预测结果
            predicted_label = torch.argmax(output).item()
            
            # 更新统计信息
            if predicted_label == true_label:
                correct += 1
                digit_correct[true_label] += 1
            
            digit_total[true_label] += 1
            
        # 输出当前进度
        print(f"已处理: {end_idx}/{total}, 当前准确率: {correct/end_idx:.4f}")
    
    # 计算总体准确率
    accuracy = correct / total
    
    # 计算各个数字的准确率
    digit_accuracies = [digit_correct[i] / digit_total[i] if digit_total[i] > 0 else 0 for i in range(10)]
    
    # 打印结果
    print(f"测试集总体准确率: {accuracy:.4f} ({correct}/{total})")
    print("各数字准确率:")
    for i in range(10):
        print(f"  数字 {i}: {digit_accuracies[i]:.4f} ({digit_correct[i]}/{digit_total[i]})")
    
    # 绘制各数字准确率柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), digit_accuracies)
    plt.xlabel('数字')
    plt.ylabel('准确率')
    plt.title('各数字准确率')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for i, acc in enumerate(digit_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'digit_accuracies.png'))
    print(f"各数字准确率已保存至 '{OUTPUT_DIR}/digit_accuracies.png'")
    
    return accuracy, digit_accuracies

def evaluate_subset_model(model_path, images, labels, indices, temperature=1.0, title='子集各数字准确率', save_name='subset_digit_accuracies.png'):
    """
    评估指定子集上的模型性能
    """
    weights, activation_weights, biases, device = load_model(model_path)
    correct = 0
    total = len(indices)
    digit_correct = [0] * 10
    digit_total = [0] * 10
    for idx in indices:
        x = torch.tensor(images[idx].reshape(784, 1), dtype=torch.float32, device=device)
        true_label = np.argmax(labels[idx])
        output, _, _ = neural_network(x, weights, activation_weights, biases, softplus, temperature)
        predicted_label = torch.argmax(output).item()
        if predicted_label == true_label:
            correct += 1
            digit_correct[true_label] += 1
        digit_total[true_label] += 1
    accuracy = correct / total
    digit_accuracies = [digit_correct[i] / digit_total[i] if digit_total[i] > 0 else 0 for i in range(10)]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), digit_accuracies)
    plt.xlabel('数字')
    plt.ylabel('准确率')
    plt.title(title)
    plt.xticks(range(10))
    plt.ylim(0, 1)
    for i, acc in enumerate(digit_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name))
    print(f"{title}已保存至 '{os.path.join(OUTPUT_DIR, save_name)}'")
    return accuracy, digit_accuracies

if __name__ == "__main__":
    model_path = os.path.join(OUTPUT_DIR, 'mnist_model.pth')
    
    # 测试模型在MNIST测试集上的表现
    test_model(model_path, num_samples=10)
    
    # 评估整个测试集上的性能
    evaluate_model(model_path)
    
    # 如果有自定义图像，可以取消注释以下代码进行测试
    # custom_image_path = '../test_imgs/your_image.png'
    # test_custom_image(model_path, custom_image_path) 