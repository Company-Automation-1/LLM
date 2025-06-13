import os
import sys
import numpy as np
import torch
import time
from tqdm import tqdm

# 导入MNIST数据集
sys.path.append("../../")
from dataset.mnist import load_mnist

# 导入模型相关函数
from activations import softplus, sigmoid
from networks import neural_network
from trainers import gradient

def train_single_sample_repeat(
    samples=10,
    epochs=1,
    batch_size=1,
    learning_rate=0.01,
    hidden_size=10,
    repeat_times=20,
    max_power=2,
    temperature=1.0
):
    print("单样本反复训练模式")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    if samples > 0 and samples < len(train_images):
        indices = np.random.choice(len(train_images), samples, replace=False)
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    print(f"训练集大小: {len(train_images)} 样本")
    print(f"测试集大小: {len(test_images)} 样本")

    input_size = 784
    output_size = 10

    # 初始化网络参数
    torch.manual_seed(42)
    weights = [
        torch.randn(hidden_size, input_size, device=device) * 0.01,
        torch.randn(output_size, hidden_size, device=device) * 0.01
    ]
    activation_weights = [
        torch.zeros((hidden_size, max_power), device=device),
        torch.zeros((output_size, max_power), device=device)
    ]
    biases = [
        torch.zeros(hidden_size, 1, device=device),
        torch.zeros(output_size, 1, device=device)
    ]

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        indices = np.random.permutation(len(train_images))
        correct_predictions = 0
        epoch_losses = []
        
        for idx in tqdm(indices):
            x = torch.tensor(train_images[idx].reshape(input_size, 1), dtype=torch.float32, device=device)
            t = torch.tensor(train_labels[idx].reshape(output_size, 1), dtype=torch.float32, device=device)
            true_label = torch.argmax(t).item()
            
            # 打印当前样本信息
            print(f"\n处理样本 {idx}:")
            print(f"真实标签: {true_label}")
            
            # 单个样本反复训练repeat_times次
            sample_losses = []
            for repeat_idx in range(repeat_times):
                output, vectors, intermediate_vectors = neural_network(
                    x, weights, activation_weights, biases, softplus, temperature
                )
                pred_label = torch.argmax(output).item()
                loss = torch.sum((output - t) * (output - t))
                sample_losses.append(loss.item())
                
                # 每个样本的第一次和最后一次迭代打印详细信息
                if repeat_idx in [0, repeat_times-1]:
                    print(f"\n第 {repeat_idx+1} 次迭代:")
                    print(f"预测标签: {pred_label}")
                    print(f"Loss: {loss.item():.4f}")
                    print(f"输出分布: {output.reshape(-1).tolist()}")
                    
                    # 计算并打印梯度
                    back_weights, back_act_weights, back_biases = gradient(
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
                    
                    print("\n梯度信息:")
                    for layer in range(len(back_weights)):
                        print(f"Layer {layer+1}:")
                        print(f"  weights梯度范围: [{back_weights[layer].min().item():.4f}, {back_weights[layer].max().item():.4f}]")
                        print(f"  activation_weights梯度范围: [{back_act_weights[layer].min().item():.4f}, {back_act_weights[layer].max().item():.4f}]")
                        print(f"  biases梯度范围: [{back_biases[layer].min().item():.4f}, {back_biases[layer].max().item():.4f}]")
                
                # 更新参数
                for layer in range(len(weights)):
                    weights[layer] -= learning_rate * back_weights[layer]
                    activation_weights[layer] -= learning_rate * back_act_weights[layer]
                    biases[layer] -= learning_rate * back_biases[layer]
            
            # 打印本样本训练后的最终预测
            with torch.no_grad():
                final_output, _, _ = neural_network(x, weights, activation_weights, biases, softplus, temperature)
                final_pred = torch.argmax(final_output).item()
                if final_pred == true_label:
                    correct_predictions += 1
                print(f"\n样本 {idx} 训练完成:")
                print(f"最终预测: {final_pred}")
                print(f"Loss变化: {sample_losses[0]:.4f} -> {sample_losses[-1]:.4f}")
            
            epoch_losses.extend(sample_losses)
        
        # 计算并打印本轮训练统计信息
        train_loss = np.mean(epoch_losses)
        train_acc = correct_predictions / len(train_images)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试集评估
        test_acc = evaluate(test_images, test_labels, weights, activation_weights, biases, temperature, device)
        test_accuracies.append(test_acc)
        
        print(f"\nEpoch {epoch+1} 总结:")
        print(f"训练损失: {train_loss:.4f}")
        print(f"训练准确率: {train_acc:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print("\n参数统计:")
        for layer in range(len(weights)):
            print(f"Layer {layer+1}:")
            print(f"  weights范围: [{weights[layer].min().item():.4f}, {weights[layer].max().item():.4f}]")
            print(f"  activation_weights范围: [{activation_weights[layer].min().item():.4f}, {activation_weights[layer].max().item():.4f}]")
            print(f"  biases范围: [{biases[layer].min().item():.4f}, {biases[layer].max().item():.4f}]")

    # 保存训练结果
    save_model_results(weights, activation_weights, biases, train_losses, train_accuracies, test_accuracies)

    # ========== 训练后立即加载并推理 ===========
    from test_mnist import test_model, evaluate_model, evaluate_subset_model
    model_path = os.path.join('output', 'mnist_model.pth')
    print("\n[自动测试] 重新加载模型并在测试集上推理:")
    acc = test_model(model_path, num_samples=10, temperature=temperature)
    print(f"[自动测试] 随机10个样本准确率: {acc:.4f}")
    print(f"[自动测试] 样例推理图像已保存至 output/test_samples.png")
    # 用训练用的样本评估
    acc_train_subset, digit_accs_train_subset = evaluate_subset_model(
        model_path, train_images, train_labels, np.arange(len(train_images)), temperature=temperature,
        title='训练子集各数字准确率', save_name='train_subset_digit_accuracies.png')
    print(f"[自动测试] 训练子集准确率: {acc_train_subset:.4f}")
    print(f"[自动测试] 训练子集各数字准确率图像已保存至 output/train_subset_digit_accuracies.png")
    # 用测试集评估
    acc_all, digit_accs = evaluate_model(model_path, temperature=temperature)
    print(f"[自动测试] 测试集准确率: {acc_all:.4f}")
    print(f"[自动测试] 测试集各数字准确率图像已保存至 output/digit_accuracies.png")

# 评估函数
@torch.no_grad()
def evaluate(test_images, test_labels, weights, activation_weights, biases, temperature, device):
    input_size = 784
    output_size = 10
    correct = 0
    for i in range(len(test_images)):
        x = torch.tensor(test_images[i].reshape(input_size, 1), dtype=torch.float32, device=device)
        t = torch.tensor(test_labels[i].reshape(output_size, 1), dtype=torch.float32, device=device)
        output, _, _ = neural_network(x, weights, activation_weights, biases, softplus, temperature)
        if torch.argmax(output) == torch.argmax(t):
            correct += 1
    return correct / len(test_images)

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

if __name__ == "__main__":
    # 可根据需要调整参数
    train_single_sample_repeat(
        samples=100,         # 训练样本数
        epochs=1,           # 训练轮数
        batch_size=1,       # 实际每次只用一个样本
        learning_rate=0.01, # 学习率
        hidden_size=10,     # 隐藏层大小
        repeat_times=20     # 每个样本反复训练次数
    ) 