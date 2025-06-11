import numpy as np
# 导入自定义模块，包含矩阵运算和神经网络相关函数
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid
from util import pt

from traner_test import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient



# TODO 测试函数
def test():
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

  # for i in range(a.shape[0]):
  #   for j in range(a.shape[1]):
  #     print(a[i][j], end=' ')
  #   print()

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


  # pt(ot=vectors[2])

  # demo()

  # pt('ot_weight', ot_weight)
  # pt('ot_aweight', ot_aweight)
  # pt('ot_bias', ot_bias)
