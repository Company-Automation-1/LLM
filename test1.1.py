import numpy as np

#! 1、定义向量幂次扩展函数
def vector_power(vector, max_power):
    """将向量扩展为幂次矩阵"""
    vector = np.asarray(vector).reshape(-1, 1)  # 强制转为列向量 (n, 1)，无论原始形状如何
    powers = np.arange(1, max_power + 1)  # 生成1到max_power的幂次序列
    return (vector ** powers).T  # 转置后形状 (max_power, n)

#! 2、定义前向传播函数
def propagate(vector, weights, activation_weight, biases):
    """
    计算向量在权重矩阵下的非线性传播结果
    
    参数:
        vector: numpy.ndarray, 形状为(n,1)的输入列向量
        weights: numpy.ndarray, 形状为(m,n)的权重矩阵，用于线性变换
        activation_weight: numpy.ndarray, 形状为(m,k)的激活权重矩阵，用于非线性变换
        biases: numpy.ndarray, 形状为(m,1)的偏置向量
    
    返回:
        numpy.ndarray, 形状为(m,1)的输出向量
    
    计算过程:
        1. 先进行线性变换: z = weights @ vector + biases
        2. 对z进行幂次扩展: 生成形状为(k,m)的幂次矩阵
        3. 对每个输出维度计算加权和: output[i] = sum(activation_weight[i,:] * vandermonde_matrix[:,i])
    
    示例:
        - vector = np.array([[1,2,3]]).T
        - weights = np.array([[2,2,1],[2,2,1],[1,1,1],[4,4,4]])
        - activation_weight = np.array([[4,3,5],[2,1,2],[1,2,6],[3,2,4]])
        - biases = np.array([[2,2,1,3]]).T
        - propagate(vector, weights, activation_weight, biases)
        >>>  返回各神经元的激活值组成的列向量
    """

    max_power = activation_weight.shape[1]

    vandermonde_matrix = vector_power(weights @ vector + biases, max_power)


    output_vector = np.array([])  # 先初始化一个空数组
    for i in range(activation_weight.shape[0]):
        output_vector = np.append(output_vector, activation_weight[i, :] @ vandermonde_matrix[:, i]).reshape(-1, 1)

    # output_vector = (activation_weight * vandermonde_matrix.T).sum(axis=1, keepdims=True)  # 计算激活权重与转置后的矩阵a的乘积的每一行的和，并保持维度
    # print(output_vector)  # 输出计算结果v
    
    """ 结果 """ 
    return output_vector

#! 3、定义矩阵训练函数
def train_matrices(initial_matrices, training_steps=3, learning_rate=0.9):
    """
    训练矩阵集合
    
    参数:
        initial_matrices: list[np.array] - 初始矩阵列表
        training_steps: int - 训练次数 (默认3次)
        learning_rate: float - 学习率/衰减率 (默认0.9)
        
    返回:
        list[np.array] - 训练后的矩阵列表
    """
    trained_matrices = initial_matrices.copy()
    
    for _ in range(training_steps):
        updated_matrices = []
        for matrix in trained_matrices:
            # 模拟训练过程：矩阵元素乘以学习率
            updated_matrix = matrix * learning_rate
            updated_matrices.append(updated_matrix)
        
        trained_matrices = updated_matrices
    
    return trained_matrices

#! 4、定义softmax函数
def softmax(x,temperature=0.7):
    """
    softmax函数

    参数:
        x: np.array - 输入
        temperature: float - 温度参数，默认为0.7

    返回:
        np.array
    """
    max_x = np.max(x)
    y = np.exp((x - max_x) / temperature)
    f_x = y / np.sum(y)

    # y = np.exp(x - np.max(x))
    # f_x = (y / np.sum(np.exp(x)))/temperature

    return f_x

#! 5、定义神经网络前向传播函数
def neural_network(input_vector, weights, activation_weights, biases): 
    """
    神经网络前向传播

    参数:
        input_vector: np.array - 输入向量
        weights: np.array - 权重矩阵
        activation_weights: np.array - 激活函数权重矩阵
        biases: np.array - 偏置向量

    返回:
        np.array - 输出向量
    """

    vector=propagate(input_vector, weights, activation_weights, biases)

    output_vector=softmax(vector)

    return output_vector



#TODOs######### 测试代码 ##########

#! 1、测试向量幂次扩展
def test_vector_power():
    """测试向量幂次扩展"""
    test_vector = np.array([[1, 2, 3, 3]]).T

    result = vector_power(test_vector, 2)
    print("输入向量:")
    print(test_vector)
    print("输出矩阵:")
    print(result.T)

#! 2、测试前向传播
def test_propagate():
    """测试前向传播"""
    # 测试数据
    vector = np.array([[1, 2, 3]]).T  # 输入向量
    weights = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1], [4, 4, 4]])  # 权重矩阵
    activation_weight = np.array([[4, 3, 5], [2, 1, 2], [1, 2, 6], [3, 2, 4]])  # 激活权重
    biases = np.array([[2, 2, 1, 3]]).T  # 偏置

    # 调用前向传播函数
    result = propagate(vector, weights, activation_weight, biases)
    print("前向传播结果:")
    print(result)

    # print(result[:, 1])

#! 3、测试矩阵训练
def test_matrix_training():
    """
    测试矩阵训练功能
    
    步骤:
    1. 创建测试矩阵
    2. 打印初始状态
    3. 执行训练
    4. 打印训练结果
    """
    # 1. 定义测试矩阵
    matrix_A = np.array([[1, 2]])
    matrix_B = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    
    # 2. 初始状态
    print("初始状态:")
    for i, matrix in enumerate([matrix_A, matrix_B]):
        print(f"矩阵 {i+1}:\n{matrix}\n")
    
    # 3. 执行训练
    trained = train_matrices([matrix_A, matrix_B])
    
    # 4. 训练结果
    print("\n训练后状态:")
    for i, matrix in enumerate(trained):
        print(f"矩阵 {i+1}:\n{matrix}\n")


#! 4、测试softmax函数
def test_softmax():
    x = np.array([1, 2, 3, 4])
    print(softmax(x))

#! 5、测试神经网络前向传播
def test_neural_network():
    """
    测试神经网络前向传播
    """
    # 1. 定义测试数据
    input_vector = np.array([[1, 2, 3]]).T
    weights = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1], [4, 4, 4]])
    activation_weights = np.array([[4, 3, 5], [2, 1, 2], [1, 2, 6], [3, 2, 4]])
    biases = np.array([[2, 2, 1, 3]]).T

    # 2. 调用前向传播函数
    output_vector = neural_network(input_vector, weights, activation_weights, biases)

    # 3. 打印输出结果
    print("输出向量:")
    print(output_vector)

# 执行测试
if __name__ == "__main__":
    # test_vector_power()
    # test_propagate()
    # test_matrix_training()
    # test_softmax()
    # test_neural_network()

    def demo():
        # 1. 定义输入数据
        input_vector = np.array([[1, 2, 3]]).T  # 输入列向量 (3,1)
        weights = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1], [4, 4, 4]])  # 权重矩阵 (4,3)
        activation_weights = np.array([[4, 3, 5], [2, 1, 2], [1, 2, 6], [3, 2, 4]])  # 激活权重 (4,3)
        biases = np.array([[2, 2, 1, 3]]).T  # 偏置 (4,1)

        # 2. 线性变换计算 z = Wx + b
        linear_output = weights @ input_vector + biases  # (4,3) @ (3,1) -> (4,1)

        # 3. 幂次扩展计算 (z^1, z^2, z^3)
        vandermonde_matrix = (linear_output ** np.arange(1, 4)).T  # (3,4) 矩阵

        # 4. 非线性激活计算 (逐神经元加权求和)
        neural_output = np.array([
            activation_weights[i] @ vandermonde_matrix[:, i] 
            for i in range(4)
        ]).reshape(-1, 1)  # (4,1)

        # 5. Softmax归一化
        def softmax(x):
            e_x = np.exp((x - np.max(x)) / 0.7)  # 带温度参数的温度
            return e_x / e_x.sum()

        final_output = softmax(neural_output)  # (4,1)

        # 6. 打印各步骤结果（调试用）
        # print("="*40 + " 计算流程 " + "="*40)
        # print(f"1. 线性变换结果:\n{linear_output}\n")
        # print(f"2. 幂次扩展矩阵(z^1|z^2|z^3):\n{vandermonde_matrix.T}\n")  # 转置后更易读
        # print(f"3. 神经元激活值:\n{neural_output}\n")
        print(f"4. 最终输出(softmax):\n{final_output}\n")
        # print("="*40 + " 结束 " + "="*40)

    # demo()