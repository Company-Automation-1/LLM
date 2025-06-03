import numpy as np

def elementwise_power(w, d):
    """对矩阵元素进行幂运算"""
    return w ** d

# def vector_power_expansion(v, max_power=3):
#     """将列向量扩展为幂次矩阵"""
#     powers = np.arange(1, max_power + 1)  # 生成1到max_power的幂次序列
#     return v ** powers  # 利用numpy的广播机制计算各幂次

def vector_power(vector, max_power):
    """将向量扩展为幂次矩阵"""
    vector = np.asarray(vector).reshape(-1, 1)  # 强制转为列向量 (n, 1)，无论原始形状如何
    powers = np.arange(1, max_power + 1)  # 生成1到max_power的幂次序列
    return (vector ** powers).T  # 转置后形状 (max_power, n)

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
        3. 对每个输出维度计算加权和: output[i] = sum(activation_weight[i,:] * power_matrix[:,i])
    
    示例:
        - vector = np.array([[1,2,3]]).T
        - weights = np.array([[2,2,1],[2,2,1],[1,1,1],[4,4,4]])
        - activation_weight = np.array([[4,3,5],[2,1,2],[1,2,6],[3,2,4]])
        - biases = np.array([[2,2,1,3]]).T
        - propagate(vector, weights, activation_weight, biases)
        >>>  返回各神经元的激活值组成的列向量
    """

    max_power = activation_weight.shape[1]

    power_matrix = vector_power(weights @ vector + biases, max_power)


    output_vector = np.array([])  # 先初始化一个空数组
    for i in range(activation_weight.shape[0]):
        output_vector = np.append(output_vector, activation_weight[i, :] @ power_matrix[:, i]).reshape(-1, 1)

    # output_vector = (activation_weight * power_matrix.T).sum(axis=1, keepdims=True)  # 计算激活权重与转置后的矩阵a的乘积的每一行的和，并保持维度
    # print(output_vector)  # 输出计算结果v
    
    """ 结果 """ 
    return output_vector

    


#TODOs######### 测试代码 ##########

def test_matrix_operations():
    """测试矩阵基本操作"""
    a = np.array([[1,2,3],[5,6,7],[9,10,11]])
    print("元素级幂运算结果:\n", elementwise_power(a, a))

def test_array_insertion():
    """测试数组插入操作"""
    # 测试1：沿轴1插入
    c = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    mylist = [np.insert(i, 0, np.array([10, 20]), axis=1) for i in c]
    print("沿轴1插入后的数组:\n", np.array(mylist))
    
    # 测试2：沿轴0插入
    ls = [np.insert(i, 0, np.array([100, 200, 300]), axis=0) for i in c]
    print("沿轴0插入后的数组:\n", np.array(ls))

def test_array_appending():
    """测试数组追加操作"""
    j = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    v = np.array([[100, 200, 300]])
    print("原始数组:\n", j)
    print("沿轴0追加后的数组:\n", np.append(j, v, axis=0))

def test_vector_power():
    """测试向量幂次扩展"""
    test_vector = np.array([[1, 2, 3, 3]]).T

    result = vector_power(test_vector, 2)
    print("输入向量:")
    print(test_vector)
    print("输出矩阵:")
    print(result.T)

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


if __name__ == "__main__":
    # test_matrix_operations()
    # test_array_insertion()
    # test_array_appending()
    # test_vector_power()
    test_propagate()
