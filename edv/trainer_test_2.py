import numpy as np
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid

def diff_vect_aweight(vectors, intermediate_vectors, weights, activation_weights, activation_function, diff_activation, k, l, i, j):
   """
   计算神经网络中激活权重梯度的反向传播
   
   该函数实现了基于链式法则的激活权重梯度计算，用于神经网络的反向传播算法
   通过递归方式计算从第k层到第l层的激活权重梯度传播
   
   参数说明：
   vectors: 各层输入向量列表，从输出层到输入层排序
   intermediate_vectors: 各层中间激活值向量列表（线性变换后，激活前的值）
   weights: 权重矩阵列表，从输出层到输入层排序
   activation_weights: 激活函数相关权重参数列表
   activation_function: 激活函数
   diff_activation: 激活函数导数
   k: 目标层索引（从1开始计数）
   l: 权重层索引（从1开始计数）
   i: 目标层中的神经元索引（从1开始计数）
   j: 激活权重的索引（从1开始计数）
   
   返回值：
   包含激活权重梯度的向量
   """
        
   # 转换为0-based索引（代码内部使用0开始的索引）
   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1

   # 当前层的前一层索引
   k_layer = k - 1

   # 递归终止条件：当目标层与权重所在层相邻时
   if k == l + 1:
      # 直接计算激活权重梯度，无需继续反向传播

      # 创建与当前层向量同形的零向量
      vector_1 = np.zeros_like(vectors[k])

      # 确定多项式激活函数的次数
      degree = activation_weights[k_layer].shape[1] - 1

      # 计算激活后的值，使用范德蒙德矩阵表示多项式激活
      # 这一步计算当前层神经元的激活值（多项式形式）
      activated_vector = vandermonde_matrix(
         intermediate_vectors[k_layer], 
         activation_function, 
         degree
      )

      # 将激活值填充到对应位置
      # 这对应于激活权重梯度计算中的激活值项
      vector_1[i] = activated_vector[j][i]

      # 返回激活权重梯度
      output = vector_1

   # 多层传播情况：需要通过递归传播误差信号
   else:
      # 确定多项式激活函数的次数
      degree = activation_weights[k_layer].shape[1] - 1

      # 计算中间向量的激活函数导数值（使用范德蒙德矩阵形式）
      # 这一步计算当前层神经元对其输入的敏感度
      diff_activation_vector = vandermonde_matrix(
         intermediate_vectors[k_layer], 
         diff_activation, 
         degree
      )

      # 对角化处理：将激活权重与Vandermonde矩阵进行元素级乘法
      # 计算激活函数导数与激活权重的组合效应
      vector_2 = diagonal_product(
         activation_weights[k_layer], 
         diff_activation_vector
      )

      # 递归计算深层梯度，通过矩阵乘法@传播误差信号
      # 符合链式法则的神经网络反向传播特性
      output = vector_2 * (
         weights[k_layer] @ diff_vect_aweight
         (
            vectors, 
            intermediate_vectors, 
            weights, 
            activation_weights, 
            activation_function, 
            diff_activation, 
            k,       # 保持目标层不变
            l + 1,   # 向权重层靠近一层（递归向下传播）
            i + 1,   # 调整神经元索引（转回1-based索引）
            j + 1    # 调整激活权重索引（转回1-based索引）
         ))

   return output
