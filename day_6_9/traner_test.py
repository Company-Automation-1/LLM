import numpy as np
import copy
# 导入自定义模块，包含矩阵运算和神经网络相关函数
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid
from util import pt

#TODO 反向传播

def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):
 
   # 转换为0-based索引
   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1
   
   # 当前层的前一层索引
   k_layer = k - 1

   # 获取激活函数的多项式次数（根据参数数量计算）
   degree = activation_weights[k_layer].shape[1] - 1  # Vandermonde矩阵的阶数

   # 计算中间向量的激活函数导数值（范德蒙德矩阵形式）
   diff_activation_vector = vandermonde_matrix(
      intermediate_vectors[k_layer],  # 当前层的中间向量
      diff_activation,               # 激活函数的导数
      degree                         # 多项式次数
   )

   # 对角化处理：将激活权重与Vandermonde矩阵进行元素级乘法
   vector_2 = diagonal_product(
      activation_weights[k_layer],   # 当前层的激活函数参数
      diff_activation_vector         # 激活函数导数矩阵
   )

   # 递归终止条件：当目标层与权重所在层相邻时
   if k == l + 1:
      # 创建与当前层向量同形的零向量
      vector_1 = np.zeros_like(vectors[k])

      # 在目标神经元位置填充前一层对应神经元的激活值
      vector_1[i] = vectors[l][j]

      # 计算梯度
      output = vector_2 * vector_1

   # 多层传播情况
   else:  
      # 递归计算深层梯度，通过矩阵乘法@传播误差信号
      output = vector_2 * (
         weights[k_layer]  @ diff_vect_weight(
            vectors, 
            intermediate_vectors, 
            weights, 
            activation_weights, 
            diff_activation, 
            k,       # 保持目标层不变
            l + 1,   # 向权重层靠近一层
            i + 1,   # 调整神经元索引
            j + 1    # 调整权重索引
         )
      )

   return output

def diff_vect_aweight(vectors, intermediate_vectors, weights, activation_weights, activation_function, diff_activation, k, l, i, j):
        
   # 转换为0-based索引
   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1

   # 当前层的前一层索引
   k_layer = k - 1

   if k == l + 1:

      vector_1 = np.zeros_like(vectors[k])

      degree = activation_weights[k_layer].shape[1] - 1

      activated_vector = vandermonde_matrix(intermediate_vectors[k_layer], activation_function, degree)

      vector_1[i] = activated_vector[j][i]

      output = vector_1

   else:

      degree = activation_weights[k_layer].shape[1] - 1

      diff_activation_vector = vandermonde_matrix(intermediate_vectors[k_layer], diff_activation, degree)

      vector_2 = diagonal_product(activation_weights[k_layer], diff_activation_vector)

      output = vector_2 * (
         weights[k_layer] @ diff_vect_aweight
         (
            vectors, 
            intermediate_vectors, 
            weights, 
            activation_weights, 
            activation_function, 
            diff_activation, 
            k, l + 1, i + 1, j + 1
         ))

   return output

def diff_vect_bias(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, j):

   # 转换为0-based索引
   k = k - 1
   l = l - 1
   j = j - 1

   # 当前层的前一层索引
   k_layer = k - 1

   degree = activation_weights[k_layer].shape[1] - 1

   diff_activation_vector = vandermonde_matrix(intermediate_vectors[k_layer], diff_activation, degree)

   vector_2 = diagonal_product(activation_weights[k_layer], diff_activation_vector)
   
   if k == l + 1:

      vector_1 = np.zeros_like(vectors[k])

      vector_1[j] = vector_2[j]

      output = vector_1

   else:

      output = vector_2 * (
         weights[k_layer] @ diff_vect_bias
         (
            vectors, 
            intermediate_vectors, 
            weights, 
            activation_weights, 
            diff_activation, 
            k, l + 1, j + 1
         ))

   return output

def gradient(vectors, intermediate_vectors, weights, activation_weights, biases, activation_function, diff_activation, actual):
  """
  name: gradient
    Param:
      vectors: 各层输入向量列表
      intermediate_vectors: 中间激活值向量列表
      weights: 权重矩阵列表
      activation_weights: 激活函数相关权重
      diff_activation: 激活函数导数
      layer, row, col: 权重索引（从1开始）
    
    Return:
      back_weights
      back_act_weights
      back_biases
  """


  back_weights = copy.deepcopy(weights)
  back_act_weights = copy.deepcopy(activation_weights)
  back_biases = copy.deepcopy(biases)

  layers = len(weights)

  for layer in range(layers):
      
      rows = weights[layer].shape[0]
      cols = weights[layer].shape[1]
      
      # 层内
      for row in range(rows):

          
         for col in range(cols):
             
            back_weights[layer][row][col] =  2 * ((vectors[layers] - actual).T @ diff_vect_weight(
               vectors, 
               intermediate_vectors, 
               weights, 
               activation_weights, 
               diff_activation, 
               len(vectors),
               layer + 1, 
               row + 1, 
               col + 1
            ))

      rows = activation_weights[layer].shape[0]
      cols = activation_weights[layer].shape[1]

      for row in range(rows):

         for col in range(cols):

            back_act_weights[layer][row][col] =  2 * ((vectors[layers] - actual).T @ diff_vect_aweight(
               vectors, 
               intermediate_vectors, 
               weights, 
               activation_weights,
               activation_function,
               diff_activation, 
               len(vectors),
               layer + 1, 
               row + 1, 
               col + 1 
            ))
      
      rows = biases[layer].shape[0]

      for row in range(rows):

         back_biases[layer][row] = 2 * ((vectors[layers] - actual).T @ diff_vect_bias(
         vectors, 
         intermediate_vectors, 
         weights, 
         activation_weights, 
         diff_activation, 
         len(vectors),
         layer + 1, 
         row + 1
      ))
      
      

  back_weights
  back_act_weights
  back_biases

  return {
     "back_weights" : back_weights,
     "back_act_weights" : back_act_weights,
     "back_biases" : back_biases
  }

