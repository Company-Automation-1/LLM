import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在导入相对于项目根目录的模块
from src.networks.core import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from src.activations.activation_functions import softplus, sigmoid
from src.utils.print_utils import pt
from src.trainers.backprop import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient

def test():
    """
    创建测试数据，生成用于测试反向传播算法的神经网络结构和数据
    
    返回:
    - vectors: 各层输入向量列表，从输出层到输入层排序
    - intermediate_vectors: 各层中间激活值向量列表
    - weights: 权重矩阵列表，从输出层到输入层排序
    - activation_weights: 激活权重列表，从输出层到输入层排序
    - biases: 偏置向量列表，从输出层到输入层排序
    """
    # 定义神经网络参数
    # 测试数据定义
    # 各层输入向量（转置为列向量）
    vector_1 = np.array([
        [1, 1, 1, 1, 1]
    ]).T
    vector_2 = np.array([
        [1, 2, 3, 4]
    ]).T
    vector_3 = np.array([
        [4, 5, 6]
    ]).T

    # 中间激活值向量
    intermediate_vector_1 = np.array([
        [5, 5, 5, 5, 5]
    ]).T
    intermediate_vector_2 = np.array([
        [5, 5, 5, 5]
    ]).T

    # 权重矩阵定义（5x4 -> 4x3）
    weight_1 = np.array([
        [2, 3, 3, 2],
        [4, 6, 2, 5],
        [7, 4, 6, 2],
        [8, 8, 8, 4],
        [9, 5, 2, 1]
    ])
    weight_2 = np.array([
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]
    ])

    # 激活函数权重参数
    activation_weight_1 = np.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2]
    ])
    activation_weight_2 = np.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2]
    ])

    biases_1 = np.array([[3, 3, 3, 3, 3]]).T
    biases_2 = np.array([[3, 3, 3, 3]]).T

    # 数据结构组织
    vectors = [vector_3/3, vector_2, vector_1]  # 从输出层到输入层
    intermediate_vectors = [intermediate_vector_2, intermediate_vector_1]
    weights = [weight_2/3, weight_1/3]  # 权重矩阵顺序调整
    activation_weights = [activation_weight_2/3, activation_weight_1/3]
    biases = [biases_2/3, biases_1/3]  # 偏置项顺序调整

    return vectors, intermediate_vectors, weights, activation_weights, biases

def demo():
    """
    简单的矩阵操作演示函数
    """
    a = np.array([
        [2, 3, 3, 2],
        [4, 6, 2, 5],
        [7, 4, 6, 2],
        [8, 8, 8, 4],
        [9, 5, 2, 1]
    ])
    a1 = np.array([
        [2, 3, 3, 2],
        [4, 6, 2, 5],
        [7, 4, 6, 2],
        [8, 8, 8, 4],
        [9, 5, 2, 1]
    ])
    a2 = np.array([
        [2, 3, 3, 2],
        [4, 6, 2, 5],
        [7, 4, 6, 2],
        [8, 8, 8, 4],
        [9, 5, 2, 1]
    ])
    a3 = np.array([
        [2, 3, 3, 2],
        [4, 6, 2, 5],
        [7, 4, 6, 2],
        [8, 8, 8, 4],
        [9, 5, 2, 1]
    ])

    aa = [a, a1, a2, a3]

    print(len(aa))

if __name__ == '__main__':
    # 执行梯度计算（从第3层到第1层，特定神经元路径）
    # ot_weight = diff_vect_weight(*test(), softplus, 3, 1, 1, 2)
    # ot_aweight = diff_vect_aweight(*test(), softplus, sigmoid, 3, 1, 1, 2)
    # ot_bias = diff_vect_bias(*test(), softplus, 3, 1, 2)

    vectors, intermediate_vectors, weights, activation_weights, biases = test()

    actual = np.array([
        [2, 2, 2, 2, 2]
    ]).T

    ot_gradient = gradient(
        vectors,
        intermediate_vectors,
        weights,
        activation_weights,
        biases,
        activation_function = softplus, 
        diff_activation = sigmoid,
        temperature = 1,
        actual = actual
    )

    pt('back_weights', ot_gradient['back_weights'])
    pt('back_act_weights', ot_gradient['back_act_weights'])
    pt('back_biases', ot_gradient['back_biases'])

    # ot_neural_network = neural_network(vectors[0], weights, activation_weights, biases, softplus, 1)
    # pt('ot_neural_network', ot_neural_network)

    # demo()
    # pt('ot_weight', ot_weight)
    # pt('ot_aweight', ot_aweight)
    # pt('ot_bias', ot_bias) 