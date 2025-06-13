import numpy as np

#TODO: 模板函数
def power(value, degree):
    """
    计算数组的幂次方
    
    参数:
    - value: 输入数组
    - degree: 幂次
    
    返回:
    - value^degree: 幂运算结果
    """
    return value ** degree  # 非0次幂返回幂次计算结果

#TODO: 生成激活函数的范德蒙德矩阵
def vandermonde_matrix(vector, function, degree):
    
    matrix = vector
    
    # 逐次添加不同幂次的激活函数值
    for i in range(degree):
        matrix = np.append(matrix, function(vector, i+1), axis=1)
    
    return matrix.T  # 转置返回，使每行表示一个幂次

#TODO: 计算两个矩阵的对角元素乘积
def diagonal_product(matrix_1, matrix_2):

    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    
    m, n = matrix_1.shape
    n2, k = matrix_2.shape
    
    # 验证中间维度是否匹配
    if n != n2:
        raise ValueError(f"维度不匹配: matrix_1 列数 ({n}) 不等于 matrix_2 行数 ({n2})")
    
    min_dim = min(m, k)
    result = np.zeros((min_dim, 1))  # 结果列向量
    
    # 计算每个神经元的激活输出
    for i in range(min_dim):
        row = matrix_1[i, :]    # 取matrix_1的第i行（激活权重）
        col = matrix_2[:, i]    # 取matrix_2的第i列（激活值）
        result[i, 0] = np.sum(row * col)  # 点乘求和
    
    return result

#TODO: 神经网络单层前向传播
def propagate(vector, 
              weights, 
              activation_weights, 
              biases, 
              activation_function):

    # 计算线性变换 z = Wx + b
    intermediate_vector = weights @ vector + biases

    # 确定激活函数的最高次幂（基于activation_weights的列数）
    degree = activation_weights.shape[1] - 1

    # 生成激活后的范德蒙德矩阵 [f(z), f(z)^1, f(z)^2, ..., f(z)^degree]
    activated_vector = vandermonde_matrix(intermediate_vector, activation_function, degree)

    # 计算对角乘积，应用激活权重
    result = diagonal_product(activation_weights, activated_vector)

    return result, intermediate_vector

#TODO: Softmax激活函数
def softmax(x, temperature):

    # 数值稳定处理 - 减去最大值防止指数溢出
    max_x = np.max(x)
    scaled_x = (x - max_x) / temperature
    exp_x = np.exp(scaled_x)
    return exp_x / np.sum(exp_x)  # 归一化

#TODO: Softmax的微分
def diff_softmax(vector, diff_vector, temperature):

    y = softmax(vector, temperature)

    output = (y * diff_vector - y * y / np.exp(vector / temperature) * (np.exp(vector / temperature).T @ diff_vector)) / temperature

    return output

#TODO: 多层神经网络前向传播
def neural_network(vector,
                   weights, 
                   activation_weights, 
                   biases, 
                   activation_function, 
                   temperature):

    next_vector = vector

    vectors = [vector]  # 保存中间向量
    intermediate_vectors = []

    # 逐层前向传播
    for k in range(len(weights)):

        next_vector, intermediate_vector = propagate(
            next_vector, 
            weights[k],
            activation_weights[k], 
            biases[k],
            activation_function
        )

        vectors.append(next_vector)
        intermediate_vectors.append(intermediate_vector)
    
    # 应用softmax得到最终输出概率分布
    output_vector = softmax(next_vector, temperature)
    
    return output_vector, vectors, intermediate_vectors 