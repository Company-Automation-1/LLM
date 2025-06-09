import numpy as np

#TODO: 激活函数
def functions(value, degree):
    if degree == 0:
        return value  # 0次幂返回原值
    else:
        return value ** degree  # 非0次幂返回幂次计算结果

#TODO: Vandermonde矩陣
def vector_power(vector, max_power):
    """
    生成范德蒙德矩阵
    
    参数:
        vector: 输入向量 (一维数组或可转换为向量的结构)
        max_power: 矩阵的最高幂次
        
        返回:
        范德蒙德矩阵 (形状: n × (max_power+1))
    """
    
    # 确保输入是一维数组
    vector_1d = np.asarray(vector).flatten()
    
    # 生成范德蒙德矩阵，幂次从低到高排列
    # 矩阵形状: n × (max_power+1)，包含0次幂到max_power次幂
    vandermonde = np.vander(vector_1d, N=max_power+1, increasing=True)
    
    # 移除全为1的0次幂列（第一列）
    output = vandermonde[:, 1:]
    # print(f"output:\n{output}")

    return output

#TODO: 生成激活范德蒙德矩阵
def vandermonde_matrix(vector, functions):
    """
    生成激活范德蒙德矩阵
    
    参数:
        vector: 范德蒙德矩阵 (形状: n × (max_power+1))
        functions: 激活函数 
    
    返回:
        转置后的激活范德蒙德矩阵 (形状: max_power × n)
    """
    # 对矩阵的每个元素应用激活函数
    rows, cols = vector.shape
    power_vandermonde = np.zeros_like(vector)
    for i in range(rows):
        for j in range(cols):
            # print(f"i={i}, j={j}, vandermonde[i, j]={vandermonde[i, j]}")
            # j从0开始对应1次幂，j+1才是当前元素的幂次
            power_vandermonde[i, j] = functions(vector[i, j], j+1)
    # print(f"power_vandermonde:\n{power_vandermonde}")
    return power_vandermonde.T  # 返回转置矩阵

#TODO: 矩阵对角元素乘积求和
def diagonal_product(matrix_1, matrix_2):
    """
    计算两个矩阵的对角元素乘积之和
    
    操作说明:
        对于结果矩阵中的每个位置 i:
        result[i] = matrix_1的第i行 点乘 matrix_2的第i列
    
    参数:
        matrix_1: 二维数组 (m × n)
        matrix_2: 二维数组 (n × k)
    
    返回:
        列向量 (min(m, k) × 1)
    """
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    
    m, n = matrix_1.shape
    n2, k = matrix_2.shape
    
    # 验证中间维度是否匹配
    if n != n2:
        raise ValueError(f"维度不匹配: matrix_1 列数 ({n}) 不等于 matrix_2 行数 ({n2})")
    
    min_dim = min(m, k)
    result = np.zeros((min_dim, 1))  # 结果列向量
    
    for i in range(min_dim):
        row = matrix_1[i, :]    # 取matrix_1的第i行
        col = matrix_2[:, i]    # 取matrix_2的第i列
        result[i, 0] = np.sum(row * col)  # 点乘求和
    
    return result

#TODO: Propagate函数
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
    # 计算线性变换
    intermediate_vector = weights @ vector + biases
    
    # 确定激活函数的最高次幂（基于activation_weights的列数）
    max_power = activation_weights.shape[1]

    vector = vector_power(intermediate_vector, max_power)  # 生成范德蒙德矩阵
    # print(f"vandermonde:\n{vandermonde}")
    
    # 生成激活后的范德蒙德矩阵
    activated_vector = vandermonde_matrix(vector, functions)
    
    # 计算对角乘积
    result = diagonal_product(activation_weights, activated_vector)
    
    return result

#TODO: Softmax函数
def softmax(x, temperature=1):
    """
    带温度参数的Softmax函数
    
    参数:
        x: 输入向量
        temperature: 温度参数 (控制输出分布的平滑度)
    
    返回:
        概率分布向量
    """
    # 数值稳定处理 - 减去最大值防止指数溢出
    max_x = np.max(x)
    scaled_x = (x - max_x) / temperature
    exp_x = np.exp(scaled_x)
    return exp_x / np.sum(exp_x)  # 归一化

#TODO: 损失函数 (平方损失函数)
def quadratic_loss_function(prediction, actual):
    """
    平方损失函数
    参数:
        prediction: 预测值（可以是标量或numpy数组）
        actual: 实际值（与prediction形状相同）
    返回:
        平方损失值
    """
    return np.sum((prediction - actual) ** 2)

#TODO: 神经网络
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
