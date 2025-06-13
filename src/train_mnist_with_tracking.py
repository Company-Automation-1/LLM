#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import functools
import pandas as pd
from collections import defaultdict
import argparse

# 导入MNIST数据集
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mnist import load_mnist

# 导入模型相关函数
from activations import softplus, sigmoid
from networks import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from trainers import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient

# 定义全局变量
OUTPUT_DIR = './output'

# 性能追踪器
class PerformanceTracker:
    def __init__(self):
        # 存储各函数的调用次数
        self.call_counts = defaultdict(int)
        # 存储各函数的总执行时间
        self.execution_times = defaultdict(float)
        # 存储所有函数的执行记录（时间戳、执行时间）
        self.execution_records = defaultdict(list)
        # 记录所有被追踪的函数
        self.tracked_functions = set()
    
    def track(self, func):
        """装饰器：追踪函数执行时间和调用次数"""
        self.tracked_functions.add(func.__name__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.call_counts[func.__name__] += 1
            self.execution_times[func.__name__] += execution_time
            self.execution_records[func.__name__].append((time.time(), execution_time))
            
            return result
        
        return wrapper
    
    def get_report(self):
        """获取性能报告数据"""
        data = []
        for func_name in self.tracked_functions:
            calls = self.call_counts[func_name]
            total_time = self.execution_times[func_name]
            avg_time = total_time / calls if calls > 0 else 0
            
            data.append({
                "Function": func_name,
                "Calls": calls,
                "TotalTime(sec)": total_time,
                "AvgTime(sec)": avg_time,
                "Percentage": 0  # 稍后计算
            })
        
        # 计算总时间和百分比
        df = pd.DataFrame(data)
        total_exec_time = df["TotalTime(sec)"].sum()
        if total_exec_time > 0:
            df["Percentage"] = (df["TotalTime(sec)"] / total_exec_time) * 100
        
        # 按总执行时间排序
        df = df.sort_values("TotalTime(sec)", ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_performance(self, save_path=None):
        """生成性能图表"""
        report = self.get_report()
        
        # 创建性能概览图表
        plt.figure(figsize=(14, 10))
        
        # 1. 执行时间饼图
        plt.subplot(2, 2, 1)
        plt.pie(
            report["TotalTime(sec)"].head(5),
            labels=report["Function"].head(5),
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title('Top 5 Functions - Time Percentage')
        
        # 2. 调用次数条形图
        plt.subplot(2, 2, 2)
        bars = plt.bar(report["Function"].head(10), report["Calls"].head(10))
        plt.title('Top 10 Functions - Call Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 3. 平均执行时间条形图
        plt.subplot(2, 2, 3)
        plt.bar(report["Function"].head(10), report["AvgTime(sec)"].head(10))
        plt.title('Top 10 Functions - Avg Execution Time (sec)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 4. 总执行时间条形图
        plt.subplot(2, 2, 4)
        plt.bar(report["Function"].head(10), report["TotalTime(sec)"].head(10))
        plt.title('Top 10 Functions - Total Execution Time (sec)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"Performance chart saved to: {save_path}")
        
        return plt.gcf()

# 创建性能追踪器实例
tracker = PerformanceTracker()

# 使用装饰器追踪关键函数的性能
# 只追踪最关键的两个函数，减少性能开销
neural_network_tracked = tracker.track(neural_network)
gradient_tracked = tracker.track(gradient)

# 注释掉其他函数的追踪，减少开销
# diff_vect_weight_tracked = tracker.track(diff_vect_weight)
# diff_vect_aweight_tracked = tracker.track(diff_vect_aweight)
# diff_vect_bias_tracked = tracker.track(diff_vect_bias)
# propagate_tracked = tracker.track(propagate)
# softmax_tracked = tracker.track(softmax)
# power_tracked = tracker.track(power)
# vandermonde_matrix_tracked = tracker.track(vandermonde_matrix)
# diagonal_product_tracked = tracker.track(diagonal_product)

def train_network(max_samples=None, batch_size=64, epochs=5, initial_learning_rate=0.01, hidden_size=100):
    """
    训练神经网络模型
    
    参数：
    max_samples: 最大训练样本数，如果为None则使用全部数据集
    batch_size: 批次大小
    epochs: 训练轮数
    initial_learning_rate: 初始学习率
    hidden_size: 隐藏层大小
    """
    print("开始训练神经网络...")
    
    # 对于小样本量自动调整隐藏层大小，避免过度计算
    if max_samples and max_samples <= 50 and hidden_size > 20:
        original_hidden_size = hidden_size
        hidden_size = 20
        print(f"小样本训练：自动降低隐藏层大小从 {original_hidden_size} 到 {hidden_size}，以加快训练速度")
    
    # 检查是否可用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    
    # 控制数据集大小
    if max_samples and max_samples < len(train_images):
        print(f"限制训练集大小为 {max_samples} 样本（原始: {len(train_images)}）")
        # 随机选择max_samples个样本
        indices = np.random.choice(len(train_images), max_samples, replace=False)
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    # 打印训练集大小
    print(f"训练集大小: {len(train_images)} 样本")
    print(f"测试集大小: {len(test_images)} 样本")
    print(f"输入图像形状: {train_images[0].shape}")
    
    # 计算训练集的内存占用
    train_memory = train_images.nbytes / (1024 * 1024)
    print(f"训练集内存占用: {train_memory:.2f} MB")
    
    # 设置超参数
    input_size = 784
    output_size = 10
    max_power = 2  # 多项式激活函数的最高次数
    temperature = 1.0  # softmax温度参数
    
    # 打印训练参数
    print(f"批次大小: {batch_size}")
    print(f"初始学习率: {initial_learning_rate}")
    print(f"训练轮数: {epochs}")
    print(f"网络结构: 输入层({input_size}) -> 隐藏层({hidden_size}) -> 输出层({output_size})")
    print(f"多项式激活函数最高次数: {max_power}")
    print(f"Softmax温度参数: {temperature}")
    
    # 计算每轮迭代次数和总迭代次数
    num_train = len(train_images)
    iter_per_epoch = max(num_train // batch_size, 1)
    total_iter = epochs * iter_per_epoch
    print(f"每轮迭代次数: {iter_per_epoch}")
    print(f"总迭代次数: {total_iter}")
    
    # 初始化网络参数
    torch.manual_seed(42)  # 设置随机种子以便复现结果
    
    # 创建两层网络结构
    weights = [
        torch.randn(hidden_size, input_size, device=device) * 0.01,  # 第一层权重
        torch.randn(output_size, hidden_size, device=device) * 0.01  # 第二层权重
    ]
    
    activation_weights = [
        torch.zeros((hidden_size, max_power), device=device),  # 第一层激活权重
        torch.zeros((output_size, max_power), device=device)   # 第二层激活权重
    ]
    
    biases = [
        torch.zeros(hidden_size, 1, device=device),  # 第一层偏置
        torch.zeros(output_size, 1, device=device)   # 第二层偏置
    ]
    
    # 计算模型参数总数
    total_params = 0
    for w in weights:
        total_params += w.numel()
    for aw in activation_weights:
        total_params += aw.numel()
    for b in biases:
        total_params += b.numel()
    
    print(f"模型参数总数: {total_params}")
    print(f"- 权重参数: {sum(w.numel() for w in weights)}")
    print(f"- 激活权重参数: {sum(aw.numel() for aw in activation_weights)}")
    print(f"- 偏置参数: {sum(b.numel() for b in biases)}")
    
    # 记录训练过程中的损失和准确率
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 训练过程
    num_train = len(train_images)
    iter_per_epoch = max(num_train // batch_size, 1)
    total_iter = epochs * iter_per_epoch
    
    # 创建一个经验回放缓冲区，用于解决灾难性遗忘问题
    replay_buffer = []
    replay_buffer_size = 1000  # 经验回放缓冲区大小
    replay_batch_size = 16     # 每次从经验回放缓冲区取出的样本数
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 计算当前学习率（学习率衰减）
        learning_rate = initial_learning_rate * (0.95 ** epoch)
        print(f"当前学习率: {learning_rate:.6f}")
        
        # 打乱训练数据
        indices = np.random.permutation(num_train)
        
        batch_losses = []
        correct_predictions = 0
        
        # 按批次训练
        for i in tqdm(range(0, num_train, batch_size)):
            batch_indices = indices[i:i+batch_size]
            batch_size_actual = len(batch_indices)  # 最后一个批次可能不足batch_size
            
            # 批次损失
            batch_loss = 0
            
            # 对批次中的每个样本进行训练
            for idx in batch_indices:
                # 获取输入和标签
                x = torch.tensor(train_images[idx].reshape(input_size, 1), dtype=torch.float32, device=device)
                t = torch.tensor(train_labels[idx].reshape(output_size, 1), dtype=torch.float32, device=device)
                
                # 前向传播
                output, vectors, intermediate_vectors = neural_network_tracked(
                    x, weights, activation_weights, biases, softplus, temperature
                )
                
                # 计算损失
                loss = torch.sum((output - t) * (output - t))
                batch_loss += loss.item()
                
                # 统计预测正确的样本数
                if torch.argmax(output) == torch.argmax(t):
                    correct_predictions += 1
                
                # 计算梯度
                back_weights, back_act_weights, back_biases = gradient_tracked(
                    vectors=vectors,
                    intermediate_vectors=intermediate_vectors,
                    weights=weights,
                    activation_weights=activation_weights,
                    biases=biases,
                    activation_function=softplus,
                    diff_activation=sigmoid,
                    temperature=temperature,
                    actual=t
                )
                
                # 更新参数
                for layer in range(len(weights)):
                    weights[layer] -= learning_rate * back_weights[layer]
                    activation_weights[layer] -= learning_rate * back_act_weights[layer]
                    biases[layer] -= learning_rate * back_biases[layer]
                
                # 将当前样本添加到经验回放缓冲区
                if len(replay_buffer) < replay_buffer_size:
                    replay_buffer.append((train_images[idx], train_labels[idx]))
                else:
                    # 随机替换缓冲区中的一个样本
                    replace_idx = random.randint(0, replay_buffer_size - 1)
                    replay_buffer[replace_idx] = (train_images[idx], train_labels[idx])
            
            # 从经验回放缓冲区中随机选择样本进行训练，防止灾难性遗忘
            if len(replay_buffer) > replay_batch_size:
                replay_indices = random.sample(range(len(replay_buffer)), replay_batch_size)
                
                for r_idx in replay_indices:
                    r_img, r_label = replay_buffer[r_idx]
                    
                    # 获取输入和标签
                    x = torch.tensor(r_img.reshape(input_size, 1), dtype=torch.float32, device=device)
                    t = torch.tensor(r_label.reshape(output_size, 1), dtype=torch.float32, device=device)
                    
                    # 前向传播
                    output, vectors, intermediate_vectors = neural_network_tracked(
                        x, weights, activation_weights, biases, softplus, temperature
                    )
                    
                    # 计算梯度
                    back_weights, back_act_weights, back_biases = gradient_tracked(
                        vectors=vectors,
                        intermediate_vectors=intermediate_vectors,
                        weights=weights,
                        activation_weights=activation_weights,
                        biases=biases,
                        activation_function=softplus,
                        diff_activation=sigmoid,
                        temperature=temperature,
                        actual=t
                    )
                    
                    # 更新参数 (使用较小的学习率来回顾历史样本)
                    replay_lr = learning_rate * 0.5
                    for layer in range(len(weights)):
                        weights[layer] -= replay_lr * back_weights[layer]
                        activation_weights[layer] -= replay_lr * back_act_weights[layer]
                        biases[layer] -= replay_lr * back_biases[layer]
            
            # 记录批次平均损失
            batch_losses.append(batch_loss / batch_size_actual)
        
        # 计算训练集上的平均损失和准确率
        train_loss = np.mean(batch_losses)
        train_accuracy = correct_predictions / num_train
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 在测试集上评估模型
        test_accuracy = evaluate(test_images, test_labels, weights, activation_weights, biases, temperature, device)
        test_accuracies.append(test_accuracy)
        
        # 计算当前epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.4f}, 测试准确率: {test_accuracy:.4f}")
        print(f"Epoch耗时: {epoch_time:.2f}秒")
        print(f"经验回放缓冲区大小: {len(replay_buffer)}")
        
        # 打印一些权重和激活权重的统计信息
        for i, w in enumerate(weights):
            print(f"第{i+1}层权重范围: [{w.min().item():.4f}, {w.max().item():.4f}], 均值: {w.mean().item():.4f}")
    
    # 保存训练结果
    save_model_results(weights, activation_weights, biases, train_losses, train_accuracies, test_accuracies)
    
    # 生成并保存性能报告
    generate_performance_report()
    
    return weights, activation_weights, biases

def evaluate(test_images, test_labels, weights, activation_weights, biases, temperature, device):
    """
    在测试集上评估模型
    """
    correct = 0
    total = len(test_images)
    
    # 为了加速评估，每次只评估一部分测试集
    eval_size = min(1000, total)
    indices = np.random.choice(total, eval_size, replace=False)
    
    for idx in indices:
        x = torch.tensor(test_images[idx].reshape(784, 1), dtype=torch.float32, device=device)
        t = torch.tensor(test_labels[idx].reshape(10, 1), dtype=torch.float32, device=device)
        
        # 前向传播
        output, _, _ = neural_network_tracked(
            x, weights, activation_weights, biases, softplus, temperature
        )
        
        # 检查预测是否正确
        if torch.argmax(output) == torch.argmax(t):
            correct += 1
    
    return correct / eval_size

def save_model_results(weights, activation_weights, biases, train_losses, train_accuracies, test_accuracies):
    """
    保存模型参数和训练结果
    """
    # 调试输出
    print("[DEBUG] train_losses:", train_losses)
    print("[DEBUG] train_accuracies:", train_accuracies)
    print("[DEBUG] test_accuracies:", test_accuracies)
    print(f"[DEBUG] train_losses length: {len(train_losses)}")
    print(f"[DEBUG] train_accuracies length: {len(train_accuracies)}")
    print(f"[DEBUG] test_accuracies length: {len(test_accuracies)}")
    # 创建结果目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 将模型参数移动到CPU上保存
    weights_cpu = [w.cpu() for w in weights]
    activation_weights_cpu = [aw.cpu() for aw in activation_weights]
    biases_cpu = [b.cpu() for b in biases]
    
    # 打印最终模型权重信息
    print("\nFinal Model Weight Statistics:")
    for i, w in enumerate(weights_cpu):
        print(f"Layer {i+1} weights - Shape: {w.shape}, Range: [{w.min().item():.4f}, {w.max().item():.4f}], Mean: {w.mean().item():.4f}")
    
    print("\nFinal Activation Weight Statistics:")
    for i, aw in enumerate(activation_weights_cpu):
        print(f"Layer {i+1} activation weights - Shape: {aw.shape}, Range: [{aw.min().item():.4f}, {aw.max().item():.4f}], Mean: {aw.mean().item():.4f}")
    
    # 计算保存的模型大小
    model_data = {
        'weights': weights_cpu,
        'activation_weights': activation_weights_cpu,
        'biases': biases_cpu
    }
    
    # 保存模型参数
    model_path = os.path.join(OUTPUT_DIR, 'mnist_model.pth')
    torch.save(model_data, model_path)
    
    # 获取保存的模型文件大小
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"\nModel file size: {model_size:.2f} MB")
    print(f"Model saved to: {os.path.abspath(model_path)}")
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()
    
    print(f"Model and training results saved to '{OUTPUT_DIR}' directory")

def generate_performance_report():
    """生成并保存性能报告"""
    # 创建结果目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成性能报告并保存
    report = tracker.get_report()
    
    # 保存性能数据到CSV
    report.to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)
    print(f"Performance report saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'performance_report.csv'))}")
    
    # 生成并保存性能图表
    tracker.plot_performance(os.path.join(OUTPUT_DIR, 'performance_charts.png'))
    
    # 打印性能摘要
    print("\nPerformance Summary (Top 10 functions):")
    print(report.head(10).to_string(index=False))

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练MNIST神经网络并跟踪性能')
    
    # 添加命令行参数
    parser.add_argument('--samples', type=int, default=5000, 
                        help='训练样本数量，设置为-1表示使用全部数据集（默认：5000）')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小（默认：64）')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数（默认：5）')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='初始学习率（默认：0.01）')
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='隐藏层大小（默认：100）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 处理命令行参数
    max_samples = None if args.samples == -1 else args.samples
    
    # 打印参数信息
    print("训练参数:")
    print(f"- 样本数量: {'全部' if max_samples is None else max_samples}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 训练轮数: {args.epochs}")
    print(f"- 初始学习率: {args.learning_rate}")
    print(f"- 隐藏层大小: {args.hidden_size}")
    
    # 训练网络
    weights, activation_weights, biases = train_network(
        max_samples=max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_learning_rate=args.learning_rate,
        hidden_size=args.hidden_size
    )
    
    end_time = time.time()
    print(f"训练完成，总耗时: {end_time - start_time:.2f} 秒") 