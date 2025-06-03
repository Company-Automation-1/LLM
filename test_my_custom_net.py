import numpy as np
from my_custom_net import neural_network
from dataset.mnist import load_mnist

# 1. 加载MNIST测试集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. 参数初始化
input_size = 784
output_size = 10
activation_weights = np.random.randn(output_size, 4)  # 这里4是幂次数，可以随意调整
max_power = activation_weights.shape[1]  # 自动推断max_power

np.random.seed(42)
weights = np.random.randn(output_size, input_size)
biases = np.random.randn(output_size, 1)

# 3. 测试前100张图片，统计准确率和Loss
correct = 0
num_test = 100
for i in range(num_test):
    # 取出第i个测试样本，并调整为列向量
    sample = x_test[i].reshape(-1, 1)  # (784, 1)
    # 前向推理，得到输出概率分布
    output = neural_network(sample, weights, activation_weights, biases)  # (10, 1)
    # 预测类别
    pred = np.argmax(output)
    # 实际类别
    label = np.argmax(t_test[i])
    # 统计正确数量
    if pred == label:
        correct += 1
    # 计算交叉熵损失（Loss）
    # 数学公式：loss = -sum_j t_j * log(y_j + 1e-7)
    # 其中 t_j 是真实标签的one-hot向量，y_j是模型输出的概率分布
    y = output.ravel()  # (10,)
    t = t_test[i]       # (10,)
    loss = -np.sum(t * np.log(y + 1e-7))
    # 打印预测、实际、概率分布和Loss
    print(f"样本{i}: 预测={pred}, 实际={label}, 概率分布={output.ravel()}, Loss={loss:.6f}")

print(f"\n前{num_test}张图片的准确率: {correct / num_test}") 