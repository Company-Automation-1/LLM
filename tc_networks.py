import torch

#TODO: 模板函数
def power(value, degree):
    """
    计算张量的幂次方
    
    参数:
    - value: 输入张量
    - degree: 幂次
    
    返回:
    - value^degree: 幂运算结果
    """
    return value ** degree

#TODO: 生成激活函数的范德蒙德矩阵
def vandermonde_matrix(vector, function, degree):
    """
    生成激活函数的范德蒙德矩阵
    
    该函数将输入向量通过激活函数转换为多项式形式的范德蒙德矩阵
    每一列代表激活函数在不同幂次下的计算结果
    
    参数:
    - vector: 输入向量（通常是神经网络某层的中间激活值）
    - function: 激活函数（如softplus, sigmoid等）
    - degree: 多项式激活函数的最高次数
    
    返回:
    - 范德蒙德矩阵，每行对应不同幂次，每列对应输入向量的不同元素
    """
    matrix = vector
    
    # 逐次添加不同幂次的激活函数值
    for i in range(degree):
        matrix = torch.cat((matrix, function(vector, i+1)), dim=1)
    
    return matrix.t()  # 转置返回，使每行表示一个幂次

#TODO: 计算两个矩阵的对角元素乘积
def diagonal_product(matrix_1, matrix_2):
    """
    计算两个矩阵的对角元素乘积
    
    该函数实现了特殊的矩阵乘法，用于神经网络中的激活权重与激活值的组合
    每个输出元素是第一个矩阵的一行与第二个矩阵对应列的点积
    
    参数:
    - matrix_1: 第一个矩阵（通常是激活权重）
    - matrix_2: 第二个矩阵（通常是激活值的范德蒙德矩阵）
    
    返回:
    - 对角乘积结果，形状为(min(m,k), 1)的列向量
    """
    m, n = matrix_1.shape
    n2, k = matrix_2.shape
    
    # 验证中间维度是否匹配
    if n != n2:
        raise ValueError(f"维度不匹配: matrix_1 列数 ({n}) 不等于 matrix_2 行数 ({n2})")
    
    min_dim = min(m, k)
    result = torch.zeros((min_dim, 1), device=matrix_1.device)  # 结果列向量
    
    # 计算每个神经元的激活输出
    for i in range(min_dim):
        row = matrix_1[i, :]    # 取matrix_1的第i行（激活权重）
        col = matrix_2[:, i]    # 取matrix_2的第i列（激活值）
        result[i, 0] = torch.sum(row * col)  # 点乘求和
    
    return result

#TODO: 神经网络单层前向传播
def propagate(vector, 
              weights, 
              activation_weights, 
              biases, 
              activation_function):
    """
    神经网络单层前向传播
    
    实现了带有多项式激活函数的神经网络层的前向计算过程:
    1. 线性变换: z = Wx + b
    2. 生成激活多项式: [f(z), f(z)^1, f(z)^2, ..., f(z)^degree]
    3. 应用激活权重: 对多项式各项进行加权
    
    参数:
    - vector: 输入向量（上一层的输出）
    - weights: 权重矩阵
    - activation_weights: 激活函数权重（多项式激活函数的系数）
    - biases: 偏置向量
    - activation_function: 激活函数
    
    返回:
    - 经过线性变换和激活函数处理后的输出向量
    - 中间向量(线性变换后的结果)
    """
    # 计算线性变换 z = Wx + b
    intermediate_vector = torch.matmul(weights, vector) + biases

    # 确定激活函数的最高次幂（基于activation_weights的列数）
    degree = activation_weights.shape[1] - 1

    # 生成激活后的范德蒙德矩阵 [f(z), f(z)^1, f(z)^2, ..., f(z)^degree]
    activated_vector = vandermonde_matrix(intermediate_vector, activation_function, degree)

    # 计算对角乘积，应用激活权重
    result = diagonal_product(activation_weights, activated_vector)
    
    return result, intermediate_vector

#TODO: Softmax激活函数
def softmax(x, temperature):
    """
    Softmax激活函数
    
    将任意实数向量转换为概率分布，总和为1
    temperature参数控制概率分布的峰度（越小越尖锐）
    
    参数:
    - x: 输入向量
    - temperature: 温度参数，控制softmax的平滑程度
    
    返回:
    - 归一化后的概率分布向量
    """
    # 数值稳定处理
    max_x = torch.max(x)
    scaled_x = (x - max_x) / temperature
    exp_x = torch.exp(scaled_x)
    return exp_x / torch.sum(exp_x)  # 归一化

#TODO: Softmax的微分
def diff_softmax(vector, diff_vector, temperature):
    """
    Softmax的微分
    
    参数:
    - vector: 输入向量
    - diff_vector: 微分向量
    - temperature: 温度参数
    
    返回:
    - softmax的微分结果
    """
    y = softmax(vector, temperature)
    
    # 计算 y * diff_vector - y * y^T * diff_vector / temperature
    output = (y * diff_vector - y * y / torch.exp(vector / temperature) * (torch.exp(vector / temperature).t() * diff_vector)) / temperature
    
    return output

#TODO: 多层神经网络前向传播
def neural_network(vector,
                   weights, 
                   activation_weights, 
                   biases, 
                   activation_function, 
                   temperature):
    """
    多层神经网络前向传播
    
    实现了完整的多层神经网络前向计算过程，包括:
    1. 逐层线性变换和激活
    2. 最后一层应用softmax归一化
    
    该实现支持自定义多项式激活函数，通过激活权重控制每层的激活特性
    
    参数:
    - vector: 输入向量 (列向量)
    - weights: 权重矩阵列表 (每层一个)
    - activation_weights: 激活权重列表 (每层一个)
    - biases: 偏置向量列表 (每层一个)
    - activation_function: 激活函数 (函数)
    - temperature: softmax温度参数
    
    返回:
    - 经过完整网络处理后的输出向量（概率分布）
    - 各层的输入向量列表
    - 中间向量列表（线性变换后，激活前的值）
    """
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