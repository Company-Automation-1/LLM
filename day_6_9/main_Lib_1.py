import numpy as np

#TODO: 模板函数
def power(value, degree):
    return value ** degree  # 非0次幂返回幂次计算结果

#TODO: 生成激活范德蒙德矩阵
def vandermonde_matrix(vector, function, degree):
    
    matrix = vector

    for i in range(degree):
        matrix = np.append(matrix, function(vector, i+1), axis=1)
    
    return matrix

#TODO: 矩阵对角元素乘积求和
def diagonal_product(matrix_1, matrix_2):

    print(f"matrix_1:\n{matrix_1}\nmatrix_2:\n{matrix_2}")

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
def propagate(vector, weights, activation_weights, biases, activation_function):

    # 计算线性变换
    intermediate_vector = weights @ vector + biases
    
    # 确定激活函数的最高次幂（基于activation_weights的列数）
    degree = activation_weights.shape[1]

    # 生成激活后的范德蒙德矩阵
    activated_vector = vandermonde_matrix(intermediate_vector, activation_function, degree)
    
    # 计算对角乘积
    result = diagonal_product(activation_weights, activated_vector)
    
    return result

#TODO: Softmax函数
def softmax(x, temperature=1):

    # 数值稳定处理 - 减去最大值防止指数溢出
    max_x = np.max(x)
    scaled_x = (x - max_x) / temperature
    exp_x = np.exp(scaled_x)
    return exp_x / np.sum(exp_x)  # 归一化

#TODO: 神经网络
def neural_network(vector, weights, activation_weights, biases, activation_function, temperature):
    """
        vector: 输入向量 (列向量)
        weights: 权重矩阵列表 (每层一个)
        activation_weights: 激活权重列表 (每层一个)
        biases: 偏置向量列表 (每层一个)
        activation_function: 激活函数 (numpy函数)
        temperature: softmax温度参数
    """
    next_vector = vector
    
    # 逐层前向传播
    for k in range(len(weights)):
        next_vector = propagate(
            next_vector, 
            weights[k], 
            activation_weights[k], 
            biases[k],
            activation_function
        )
    
    # 应用softmax得到最终输出概率分布
    output_vector = softmax(next_vector, temperature)
    
    return output_vector
