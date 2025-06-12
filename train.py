#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入项目模块
from networks import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from activations import softplus, sigmoid
from util import pt
from trainers import gradient
from model import save_model, load_model
from mnist.dataset.mnist import load_mnist

class MnistTrainer:
    """
    MNIST数据集自动化训练器
    实现批量训练、模型保存和评估功能
    """
    def __init__(self, 
                 input_size=784, 
                 hidden_size=100, 
                 output_size=10, 
                 max_power=2,
                 learning_rate=0.01,
                 batch_size=100,
                 temperature=1.0):
        """
        初始化训练器
        
        参数:
        - input_size: 输入层大小 (MNIST为784)
        - hidden_size: 隐藏层大小
        - output_size: 输出层大小 (数字分类为10)
        - max_power: 激活函数多项式最高次数
        - learning_rate: 学习率
        - batch_size: 批量大小
        - temperature: softmax温度参数
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_power = max_power
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.temperature = temperature
        
        # 初始化网络参数
        np.random.seed(42)
        # 两层网络: 输入层 -> 隐藏层 -> 输出层
        self.weights = [
            np.random.randn(hidden_size, input_size) * 0.01,  # 第一层权重
            np.random.randn(output_size, hidden_size) * 0.01  # 第二层权重
        ]
        
        self.activation_weights = [
            np.random.randn(hidden_size, max_power) * 0.01,  # 第一层激活权重
            np.random.randn(output_size, max_power) * 0.01   # 第二层激活权重
        ]
        
        self.biases = [
            np.random.randn(hidden_size, 1) * 0.01,  # 第一层偏置
            np.random.randn(output_size, 1) * 0.01   # 第二层偏置
        ]
        
        # 训练历史记录
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        
        # 加载数据集
        (self.x_train, self.t_train), (self.x_test, self.t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
        
        # 转换标签为one-hot格式和列向量格式
        self.y_train = self._to_one_hot(self.t_train)
        self.y_test = self._to_one_hot(self.t_test)
        
        pt("数据集加载完成", f"训练集: {self.x_train.shape}, 测试集: {self.x_test.shape}")
    
    def _to_one_hot(self, labels):
        """
        将标签转换为one-hot向量
        
        参数:
        - labels: 标签数组
        
        返回:
        - one-hot向量数组
        """
        one_hot = np.zeros((labels.size, self.output_size))
        for i, label in enumerate(labels):
            one_hot[i, label] = 1
        return one_hot
    
    def _to_column_vector(self, data):
        """
        将向量转换为列向量 (N, ) -> (N, 1)
        
        参数:
        - data: 输入向量
        
        返回:
        - 列向量
        """
        return data.reshape(-1, 1)
    
    def train(self, epochs=10, verbose=True, save_interval=1, model_path='./models'):
        """
        训练网络
        
        参数:
        - epochs: 训练轮数
        - verbose: 是否显示进度
        - save_interval: 模型保存间隔（轮数）
        - model_path: 模型保存路径
        
        返回:
        - 训练历史记录
        """
        # 确保模型保存目录存在
        os.makedirs(model_path, exist_ok=True)
        
        n_samples = self.x_train.shape[0]
        n_batches = n_samples // self.batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            # 打乱数据
            indices = np.random.permutation(n_samples)
            x_train_shuffled = self.x_train[indices]
            y_train_shuffled = self.y_train[indices]
            t_train_shuffled = self.t_train[indices]
            
            # 批量训练
            batch_iter = range(n_batches)
            if verbose:
                batch_iter = tqdm(batch_iter, desc=f"Epoch {epoch+1}/{epochs}")
                
            for batch_idx in batch_iter:
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                
                # 获取当前批次数据
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                t_batch = t_train_shuffled[start_idx:end_idx]
                
                # 批量训练
                batch_loss = 0
                batch_correct = 0
                
                for i in range(self.batch_size):
                    # 获取单个样本
                    x = self._to_column_vector(x_batch[i])
                    y = self._to_column_vector(y_batch[i])
                    
                    # 前向传播
                    output, vectors, intermediate_vectors = neural_network(
                        x, 
                        self.weights, 
                        self.activation_weights, 
                        self.biases, 
                        softplus, 
                        self.temperature
                    )
                    
                    # 计算损失
                    loss = np.sum((output - y) * (output - y))
                    batch_loss += loss
                    
                    # 计算准确率
                    pred = np.argmax(output)
                    if pred == t_batch[i]:
                        batch_correct += 1
                    
                    # 反向传播
                    back_weights, back_act_weights, back_biases = gradient(
                        vectors = vectors,
                        intermediate_vectors = intermediate_vectors,
                        weights = self.weights,
                        activation_weights = self.activation_weights,
                        biases = self.biases,
                        activation_function = softplus,
                        diff_activation = sigmoid,
                        temperature = self.temperature,
                        actual = y
                    )
                    
                    # 更新参数
                    for layer in range(len(self.weights)):
                        self.weights[layer] -= self.learning_rate * back_weights[layer]
                        self.activation_weights[layer] -= self.learning_rate * back_act_weights[layer]
                        self.biases[layer] -= self.learning_rate * back_biases[layer]
                
                # 累加损失和正确预测数
                total_loss += batch_loss
                correct += batch_correct
                
                # 显示当前批次的训练情况
                if verbose:
                    batch_iter.set_postfix(
                        loss=batch_loss/self.batch_size, 
                        acc=batch_correct/self.batch_size
                    )
            
            # 计算训练集上的平均损失和准确率
            avg_loss = total_loss / n_samples
            train_acc = correct / n_samples
            
            # 在测试集上评估
            test_acc = self.evaluate()
            
            # 记录历史
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(train_acc)
            self.test_acc_history.append(test_acc)
            
            # 输出当前轮次的训练情况
            if verbose:
                pt(f"Epoch {epoch+1}/{epochs}", f"loss: {avg_loss:.6f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
            
            # 保存模型
            if (epoch + 1) % save_interval == 0:
                self.save_model(f"{model_path}/model_epoch_{epoch+1}.pkl")
        
        # 保存最终模型
        self.save_model(f"{model_path}/model_final.pkl")
        
        # 绘制训练历史图表
        self.plot_history()
        
        return {
            'train_loss': self.train_loss_history,
            'train_acc': self.train_acc_history,
            'test_acc': self.test_acc_history
        }
    
    def evaluate(self, samples=1000):
        """
        在测试集上评估模型
        
        参数:
        - samples: 评估样本数量 (为了加速，可以只评估部分样本)
        
        返回:
        - 准确率
        """
        if samples > len(self.x_test):
            samples = len(self.x_test)
            
        correct = 0
        indices = np.random.choice(len(self.x_test), samples, replace=False)
        
        for idx in indices:
            x = self._to_column_vector(self.x_test[idx])
            
            # 前向传播
            output, _, _ = neural_network(
                x, 
                self.weights, 
                self.activation_weights, 
                self.biases, 
                softplus, 
                self.temperature
            )
            
            # 计算准确率
            pred = np.argmax(output)
            if pred == self.t_test[idx]:
                correct += 1
        
        return correct / samples
    
    def predict(self, x):
        """
        预测单个样本
        
        参数:
        - x: 输入数据 (784维向量)
        
        返回:
        - 预测结果 (概率分布)
        """
        x = self._to_column_vector(x)
        
        # 前向传播
        output, _, _ = neural_network(
            x, 
            self.weights, 
            self.activation_weights, 
            self.biases, 
            softplus, 
            self.temperature
        )
        
        return output
    
    def save_model(self, filepath):
        """
        保存模型参数
        
        参数:
        - filepath: 保存路径
        """
        model_params = {
            'weights': self.weights,
            'activation_weights': self.activation_weights,
            'biases': self.biases,
            'temperature': self.temperature,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'max_power': self.max_power
        }
        
        save_model(model_params, filepath)
    
    def load_model(self, filepath):
        """
        加载模型参数
        
        参数:
        - filepath: 加载路径
        """
        model_params = load_model(filepath)
        
        self.weights = model_params['weights']
        self.activation_weights = model_params['activation_weights']
        self.biases = model_params['biases']
        self.temperature = model_params['temperature']
        self.input_size = model_params['input_size']
        self.hidden_size = model_params['hidden_size']
        self.output_size = model_params['output_size']
        self.max_power = model_params['max_power']
        
        pt("模型加载完成", filepath)
    
    def plot_history(self):
        """绘制训练历史图表"""
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history)
        plt.title('训练损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label='训练集')
        plt.plot(self.test_acc_history, label='测试集')
        plt.title('准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./results/training_history.png')
        plt.close()

def main():
    """主函数"""
    # 训练参数
    params = {
        'input_size': 784,
        'hidden_size': 100,
        'output_size': 10,
        'max_power': 2,
        'learning_rate': 0.01,
        'batch_size': 100,
        'temperature': 1.0
    }
    
    # 训练轮数
    epochs = 10
    
    # 创建训练器
    trainer = MnistTrainer(**params)
    
    # 训练模型
    trainer.train(epochs=epochs, verbose=True, save_interval=2)
    
    # 保存最终模型
    trainer.save_model('./models/final_model.pkl')
    
    pt("训练完成", f"模型已保存到 ./models/final_model.pkl")

if __name__ == "__main__":
    main() 