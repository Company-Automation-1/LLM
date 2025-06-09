import numpy as np
import test

# 激活函數
def activation_function(value, degree):
    if degree == 0:
        return value
    else:
        return value ** degree

# 生成范德蒙德矩阵并应用激活函数
def vandermonde_matrix(vector, power, activation_function=activation_function):
    """
    生成范德蒙德矩阵，并对每个元素应用激活函数。
    
    参数:
        vector: 用于生成范德蒙德矩阵的基向量。
        power: 矩阵的最高次幂。
        activation_function: 激活函数，接受两个参数 (value, degree)。
    
    返回:
        应用了激活函数的范德蒙德矩阵。
    """
    # 生成范德蒙德矩阵, 并去掉第一列（全为1）, 并通过increasing=True参数指定幂次顺序从低到高
    vandermonde = np.vander(vector, N=power+1, increasing=True)[:, 1:]
    
    # 对矩阵的每个元素应用激活函数
    rows, cols = vandermonde.shape
    activated_vandermonde = np.zeros_like(vandermonde)
    for i in range(rows):
        for j in range(cols):
            activated_vandermonde[i, j] = activation_function(vandermonde[i, j], j)
    
    return activated_vandermonde.T
# 損失函數 (平方损失函数)
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

# 對角乘積
def diagonal_product(matrix_1, matrix_2):
    """
    计算两个矩阵的「对角乘積」
    
    参数:
        matrix_1: 二维数组, 尺寸 (m, n)
        matrix_2: 二维数组, 尺寸 (n, k)
        
    返回:
        二维数组 (列向量), 尺寸 (min(m, k), 1)
    """
    # 转换为 NumPy 数组（如果输入是列表）
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    
    m, n = matrix_1.shape
    n2, k = matrix_2.shape
    
    # 检查中间维度是否匹配
    if n != n2:
        raise ValueError(f"中间维度不匹配: matrix_1 的列数 ({n}) 不等于 matrix_2 的行数 ({n2})")
    
    # 结果长度取 min(m, k)
    min_dim = min(m, k)
    result = np.zeros((min_dim, 1))  # 创建列向量
    
    for i in range(min_dim):
        # 取 matrix_1 的第 i 行 和 matrix_2 的第 i 列
        row = matrix_1[i, :]
        col = matrix_2[:, i]
        # 对应元素相乘后求和
        result[i, 0] = np.sum(row * col)
    
    return result


def test_quadratic_loss_function():
    """
    测试平方损失函数
    """
    prediction = np.array([[0, 1, 0]]).T
    actual = np.array([[1, 1, 1]]).T
    loss = quadratic_loss_function(prediction, actual)
    print("平方损失值:", loss)



def test_vandermonde_matrix():
    """
    测试范德蒙德矩阵的生成和激活。
    """

    vector = np.array([
        1, 2, 3
    ])
    power = 3

    # 调用函数
    result = vandermonde_matrix(vector, power)
    print("激活后的范德蒙德矩阵:")
    print(result)

def test_diagonal_product():
    """
    测试对角元素乘积。
    """
    matrix_1 = np.array([
        [1, 2, 3],
        [5, 5, 5]
    ])

    matrix_2 = np.array([
        [1, 1],
        [2, 1],
        [3, 1]
    ])

    result = diagonal_product(matrix_1, matrix_2)
    print(result)  # 输出: [[14.] 
                   #       [15.]]

if __name__ == '__main__':
    # test_vandermonde_matrix() # 测试Vandermonde矩阵
    # test_quadratic_loss_function() # 测试平方损失函数
    test_diagonal_product() # 测试对角元素乘积
