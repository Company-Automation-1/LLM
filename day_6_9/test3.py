import torch
import torch.nn.functional as F

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 激活函数 - 根据指定的幂次对输入值进行变换
def power(value, degree):
    if degree == 0:
        return value  # 0次幂返回原值
    elif degree == 1:
        return value  # 1次幂直接返回，避免不必要的计算
    else:
        return value ** degree  # 非0/1次幂返回幂次计算结果

# 生成激活范德蒙德矩阵
def vandermonde_matrix(vector, max_power, power):
    """
    生成激活范德蒙德矩阵
    
    参数:
        vector: 输入向量 (一维张量或可转换为向量的结构)
        max_power: 矩阵的最高幂次
        power: 激活函数 
    
    返回:
        转置后的激活范德蒙德矩阵 (形状: max_power × n)
    """
    # 边缘条件检查
    if max_power <= 0:
        raise ValueError("max_power必须大于0")
    
    # 确保输入是一维张量
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    
    # 处理空向量的情况
    if vector.numel() == 0:
        return torch.zeros((0, max_power), dtype=torch.float32, device=device)
    
    vector_1d = vector.flatten()
    
    # 使用矩阵运算优化范德蒙德矩阵构建
    rows = vector_1d.size(0)
    vandermonde = torch.zeros((rows, max_power+1), dtype=torch.float32, device=device)
    
    # 第一列全为1（0次幂）
    vandermonde[:, 0] = 1.0
    
    # 第二列为原始值（1次幂）
    if max_power >= 1:
        vandermonde[:, 1] = vector_1d
    
    # 剩余列可以通过乘法得到，减少幂运算
    for i in range(2, max_power+1):
        vandermonde[:, i] = vandermonde[:, i-1] * vector_1d
    
    # 移除全为1的0次幂列（第一列）
    vandermonde = vandermonde[:, 1:]
    print(f"vandermonde:\n{vandermonde}")
    
    # 对矩阵的每个元素应用激活函数（使用广播优化）
    rows, cols = vandermonde.shape
    power_vandermonde = torch.zeros_like(vandermonde)
    
    # 优化循环，对每列使用向量操作
    for j in range(cols):
        # j从0开始对应1次幂，j+1才是当前元素的幂次
        power_vandermonde[:, j] = power(vandermonde[:, j], j+1)
    
    print(f"power_vandermonde:\n{power_vandermonde}")
    return power_vandermonde.T  # 返回转置矩阵

# 矩阵对角元素乘积求和
def diagonal_product(matrix_1, matrix_2):
    """
    计算两个矩阵的对角元素乘积之和
    
    操作说明:
        对于结果矩阵中的每个位置 i:
        result[i] = matrix_1的第i行 点乘 matrix_2的第i列
    
    参数:
        matrix_1: 二维张量 (m × n)
        matrix_2: 二维张量 (n × k)
    
    返回:
        列向量 (min(m, k) × 1)
    """
    # 边缘条件检查
    if not isinstance(matrix_1, torch.Tensor):
        matrix_1 = torch.tensor(matrix_1, dtype=torch.float32, device=device)
    if not isinstance(matrix_2, torch.Tensor):
        matrix_2 = torch.tensor(matrix_2, dtype=torch.float32, device=device)
    
    # 处理空矩阵的情况
    if matrix_1.numel() == 0 or matrix_2.numel() == 0:
        return torch.zeros((0, 1), dtype=torch.float32, device=device)
    
    m, n = matrix_1.shape
    n2, k = matrix_2.shape
    
    # 验证中间维度是否匹配
    if n != n2:
        raise ValueError(f"维度不匹配: matrix_1 列数 ({n}) 不等于 matrix_2 行数 ({n2})")
    
    min_dim = min(m, k)
    result = torch.zeros((min_dim, 1), dtype=torch.float32, device=device)  # 结果列向量
    
    # 优化计算，避免显式循环
    for i in range(min_dim):
        result[i, 0] = torch.sum(matrix_1[i, :] * matrix_2[:, i])  # 点乘求和
    
    return result

# Propagate函數
def propagate(vector, weights, activation_weights, biases):
    """
    执行单层神经网络前向传播
    
    步骤:
    1. 线性变换: intermediate = weights @ vector + biases
    2. 生成激活范德蒙德矩阵
    3. 计算对角乘积: activation_weights 与 激活矩阵的乘积
    
    参数:
        vector: 输入列向量
        weights: 权重矩阵
        activation_weights: 激活权重矩阵
        biases: 偏置向量
    
    返回:
        该层的输出向量
    """
    # 边缘条件检查
    # 转换为张量
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if not isinstance(activation_weights, torch.Tensor):
        activation_weights = torch.tensor(activation_weights, dtype=torch.float32, device=device)
    if not isinstance(biases, torch.Tensor):
        biases = torch.tensor(biases, dtype=torch.float32, device=device)
    
    # 处理空向量/矩阵的情况
    if vector.numel() == 0 or weights.numel() == 0:
        return torch.zeros((activation_weights.shape[0], 1), dtype=torch.float32, device=device)
    
    # 维度检查
    if vector.dim() == 1:
        vector = vector.unsqueeze(1)  # 确保是列向量
    
    # 计算线性变换
    intermediate_vector = weights @ vector + biases
    
    # 确定激活函数的最高次幂（基于activation_weights的列数）
    max_power = activation_weights.shape[1]
    
    # 生成激活后的范德蒙德矩阵
    activated_vector = vandermonde_matrix(intermediate_vector, max_power, power)
    
    # 计算对角乘积
    result = diagonal_product(activation_weights, activated_vector)
    
    return result

# Softmax函数
def softmax(x, temperature=1):
    """
    带温度参数的Softmax函数
    
    参数:
        x: 输入向量
        temperature: 温度参数 (控制输出分布的平滑度)
    
    返回:
        概率分布向量
    """
    # 边缘条件检查
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    # 处理空向量的情况
    if x.numel() == 0:
        return torch.zeros((0, 1), dtype=torch.float32, device=device)
    
    # 处理temperature为0的情况（防止除零错误）
    if temperature == 0:
        temperature = 1e-8
    
    # 数值稳定处理 - 减去最大值防止指数溢出
    max_x = torch.max(x)
    scaled_x = (x - max_x) / temperature
    
    # 防止溢出处理
    scaled_x = torch.clamp(scaled_x, min=-50.0, max=50.0)
    
    exp_x = torch.exp(scaled_x)
    sum_exp = torch.sum(exp_x)
    
    # 防止除零错误
    if sum_exp == 0:
        sum_exp = 1e-8
    
    return exp_x / sum_exp  # 归一化

# 損失函數 (平方损失函数)
def quadratic_loss_function(prediction, actual):
    """
    平方损失函数
    参数:
        prediction: 预测值（可以是标量或张量）
        actual: 实际值（与prediction形状相同）
    返回:
        平方损失值
    """
    # 边缘条件检查
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction, dtype=torch.float32, device=device)
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual, dtype=torch.float32, device=device)
    
    # 处理空向量的情况
    if prediction.numel() == 0 or actual.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    
    # 确保形状匹配
    if prediction.shape != actual.shape:
        raise ValueError(f"形状不匹配: prediction {prediction.shape} 不等于 actual {actual.shape}")
    
    return torch.sum((prediction - actual) ** 2)

# 神经网络前向传播
def neural_network(vector, weights, activation_weights, biases, temperature):
    """
    多层神经网络前向传播
    
    参数:
        vector: 输入向量 (列向量)
        weights: 权重矩阵列表 (每层一个)
        activation_weights: 激活权重列表 (每层一个)
        biases: 偏置向量列表 (每层一个)
        temperature: softmax温度参数
    
    返回:
        输出概率分布 (列向量)
    """
    # 边缘条件检查
    if not weights or not activation_weights or not biases:
        raise ValueError("权重、激活权重和偏置列表不能为空")
    
    if len(weights) != len(activation_weights) or len(weights) != len(biases):
        raise ValueError("权重、激活权重和偏置列表长度必须相同")
    
    # 转换输入为张量
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    
    # 确保输入是列向量
    if vector.dim() == 1:
        vector = vector.unsqueeze(1)
    
    next_vector = vector
    
    # 逐层前向传播
    for k in range(len(weights)):
        next_vector = propagate(
            next_vector, 
            weights[k], 
            activation_weights[k], 
            biases[k]
        )
    
    # 应用softmax得到最终输出概率分布
    output_vector = softmax(next_vector, temperature)
    
    return output_vector

# 测试神经网络
def test_neural_network():
    """
    测试神经网络前向传播
    
    构造一个两层网络进行测试，验证：
    1. 输出是否为概率分布 (和为1)
    2. 温度参数对输出的影响
    """
    print(f"使用设备: {device}")
    
    # 输入向量 (3×1)
    input_vector = torch.tensor([[1, 2, 3]], dtype=torch.float32, device=device).T
    
    # 第一层参数
    weights_layer1 = torch.tensor([
        [2, 2, 1],
        [2, 2, 1],
        [1, 1, 1],
        [4, 4, 4]
    ], dtype=torch.float32, device=device)  # 4×3

    activation_weights_layer1 = torch.tensor([
        [4, 3, 5],
        [2, 1, 2],
        [1, 2, 6],
        [3, 2, 4]
    ], dtype=torch.float32, device=device)  # 4×3

    biases_layer1 = torch.tensor([[2, 2, 1, 3]], dtype=torch.float32, device=device).T  # 4×1
    
    # 组织为层参数列表
    weights = [weights_layer1]
    activation_weights = [activation_weights_layer1]
    biases = [biases_layer1]
    
    # 温度参数
    temperature = 1
    
    # 运行神经网络
    output = neural_network(input_vector, weights, activation_weights, biases, temperature)
    
    # 打印结果
    print("神经网络输出:")
    print(output)
    # print(output.cpu().detach().numpy())  # 将结果转到CPU并转为numpy打印，避免显示设备信息
    
    # # 验证输出属性
    # assert output.shape[1] == 1, "输出应为列向量"
    # assert torch.isclose(torch.sum(output), torch.tensor(1.0, device=device), atol=1e-6), "输出概率和应为1"
    
    # print("\n测试通过!")
    
    # # 测试不同温度的影响
    # print("\n测试不同温度下的输出:")
    # for temp in [0.1, 1.0, 10.0]:
    #     output_temp = neural_network(input_vector, weights, activation_weights, biases, temp)
    #     print(f"\n温度 = {temp}:")
    #     print(output_temp.cpu().detach().numpy())
    #     max_val = torch.max(output_temp).item()
    #     min_val = torch.min(output_temp).item()
    #     print(f"最大值: {max_val:.4f} 最小值: {min_val:.4f}")

# 执行测试
if __name__ == "__main__":
    test_neural_network()