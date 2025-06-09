#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from power_activation_network import PowerActivationNetwork
import os
import time
import sys
from tqdm import tqdm

def load_mnist_data():
    """
    加载MNIST数据集
    如果MNIST数据集不可用，则使用随机生成的数据进行测试
    """
    try:
        # 尝试从mnist目录导入load_mnist函数
        sys.path.append(os.path.abspath('.'))
        from dataset.mnist import load_mnist
        print("成功导入MNIST数据集加载函数")
        
        # 加载MNIST数据
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False)
        print(f"成功加载MNIST数据集: 训练集 {x_train.shape}, 测试集 {x_test.shape}")
        return (x_train, t_train), (x_test, t_test)
    
    except Exception as e:
        print(f"无法加载MNIST数据集: {str(e)}")
        print("使用随机生成的数据进行测试...")
        
        # 生成随机数据用于测试
        x_train = np.random.rand(1000, 1, 28, 28)
        t_train = np.random.randint(0, 10, 1000)
        x_test = np.random.rand(100, 1, 28, 28)
        t_test = np.random.randint(0, 10, 100)
        
        return (x_train, t_train), (x_test, t_test)

def train_network(epochs=5, batch_size=100, save_interval=1, learning_rate=0.01, hidden_size=50, max_power=3):
    """
    使用MNIST数据集训练幂激活网络
    
    参数:
    epochs: 训练轮数
    batch_size: 批处理大小
    save_interval: 每隔多少轮保存一次模型
    learning_rate: 学习率
    hidden_size: 隐藏层神经元数量
    max_power: 激活函数的最大幂次
    """
    print("开始训练幂激活神经网络...")
    
    # 加载MNIST数据集
    (x_train, t_train), (x_test, t_test) = load_mnist_data()
    
    # 创建网络
    network = PowerActivationNetwork(input_size=784, hidden_size=hidden_size, output_size=10, max_power=max_power)
    network.learning_rate = learning_rate
    print(f"创建网络: 输入层 784, 隐藏层 {hidden_size}, 输出层 10, 最大幂次 {max_power}, 学习率 {learning_rate}")
    
    # 创建保存目录
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 尝试加载已有模型
    model_path = "power_network_params.pkl"
    if os.path.exists(model_path):
        success = network.load_params(model_path)
        if success:
            print("成功加载已有模型，继续训练")
        else:
            print("模型加载失败，使用新初始化的模型")
    else:
        print("未找到预训练模型，使用新初始化的模型")
    
    # 训练参数
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    total_iterations = epochs * iter_per_epoch
    
    # 记录训练过程
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    
    print(f"开始训练: {epochs}轮, 每轮{iter_per_epoch}次迭代, 总共{total_iterations}次迭代")
    print(f"批处理大小: {batch_size}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()
        
        # 打乱训练数据
        perm = np.random.permutation(train_size)
        
        # 记录这个epoch的损失
        epoch_loss = 0
        
        # 批量处理，使用tqdm显示进度条
        pbar = tqdm(range(iter_per_epoch), desc=f"训练进度", unit="batch")
        for iteration in pbar:
            # 获取小批量数据
            batch_mask = perm[iteration*batch_size:(iteration+1)*batch_size]
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            # 批量更新网络权重
            network.update_weights(x_batch, t_batch)
            
            # 计算损失 (仅用于监控，不影响训练)
            if (iteration + 1) % 10 == 0:
                batch_loss = network.calculate_loss(x_batch, t_batch)
                epoch_loss += batch_loss
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
        
        # 计算平均epoch损失
        avg_loss = epoch_loss / (iter_per_epoch // 10)
        loss_list.append(avg_loss)
        
        # 评估模型性能
        print("  计算训练集准确率...")
        train_acc = network.evaluate(x_train[:1000], t_train[:1000], batch_size=200)
        print("  计算测试集准确率...")
        test_acc = network.evaluate(x_test, t_test, batch_size=200)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        print(f"  Epoch {epoch+1} 完成, 用时: {epoch_time:.2f}秒")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  训练集准确率: {train_acc:.4f}, 测试集准确率: {test_acc:.4f}")
        
        # 保存模型
        if (epoch + 1) % save_interval == 0:
            model_filename = os.path.join(model_dir, f"power_network_epoch{epoch+1}.pkl")
            network.save_params(model_filename)
    
    # 保存最终模型
    network.save_params()
    total_time = time.time() - start_time
    print(f"训练完成，总用时: {total_time:.2f}秒")
    print("最终模型已保存: power_network_params.pkl")
    
    # 绘制训练过程
    plot_training_history(train_acc_list, test_acc_list, loss_list, epochs)
    
    return network

def plot_training_history(train_acc_list, test_acc_list, loss_list, epochs):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    # 绘制准确率图
    plt.subplot(1, 2, 1)
    x = np.arange(1, epochs + 1)
    plt.plot(x, train_acc_list, marker='o', label='训练集准确率')
    plt.plot(x, test_acc_list, marker='s', label='测试集准确率')
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.title("幂激活网络训练准确率")
    plt.grid(True)
    
    # 绘制损失图
    plt.subplot(1, 2, 2)
    plt.plot(x, loss_list, marker='o', color='r')
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.title("幂激活网络训练损失")
    plt.grid(True)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("训练历史图已保存为 training_history.png")
    
    try:
        plt.show()
    except:
        print("无法显示图形，但已保存为文件")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练幂激活神经网络')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=100, help='批处理大小')
    parser.add_argument('--save-interval', type=int, default=1, help='保存间隔（轮）')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--hidden-size', type=int, default=50, help='隐藏层神经元数量')
    parser.add_argument('--max-power', type=int, default=3, help='激活函数最大幂次')
    
    args = parser.parse_args()
    
    # 使用参数训练
    train_network(
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        save_interval=args.save_interval,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        max_power=args.max_power
    ) 