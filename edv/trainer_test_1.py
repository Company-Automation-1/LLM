import numpy as np
# 导入自定义模块，包含矩阵运算和神经网络相关函数
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid

def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):
   """
   计算神经网络中权重梯度的反向传播
   
   该函数实现了基于链式法则的权重梯度计算，用于神经网络的反向传播算法
   通过递归方式计算从第k层到第l层的权重梯度传播
   
   参数说明：
   vectors: 各层输入向量列表，从输出层到输入层排序
   intermediate_vectors: 各层中间激活值向量列表（线性变换后，激活前的值）
   weights: 权重矩阵列表，从输出层到输入层排序
   activation_weights: 激活函数相关权重参数列表
   diff_activation: 激活函数导数
   k: 目标层索引（从1开始计数）
   l: 权重层索引（从1开始计数）
   i: 目标层中的神经元索引（从1开始计数）
   j: 权重层中的神经元索引（从1开始计数）
   
   返回值：
   包含权重梯度的向量
   """

   # 转换为0-based索引（代码内部使用0开始的索引）
   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1

   # 当前层的前一层索引
   k_layer = k - 1

   # 获取激活函数的多项式次数（根据参数数量计算）
   # 激活权重矩阵的列数减1表示多项式的最高次数
   degree = activation_weights[k_layer].shape[1] - 1  # Vandermonde矩阵的阶数

   # 计算中间向量的激活函数导数值（使用范德蒙德矩阵形式）
   # 这一步计算当前层神经元对其输入的敏感度
   diff_activation_vector = vandermonde_matrix(
      intermediate_vectors[k_layer],  # 当前层的中间向量（激活前的值）
      diff_activation,               # 激活函数的导数
      degree                         # 多项式次数
   )

   # 对角化处理：将激活权重与Vandermonde矩阵进行元素级乘法
   # 计算激活函数导数与激活权重的组合效应
   vector_2 = diagonal_product(
      activation_weights[k_layer],   # 当前层的激活函数参数
      diff_activation_vector         # 激活函数导数矩阵
   )

   # 递归终止条件：当目标层与权重所在层相邻时
   if k == l + 1:
      # 直接计算梯度，无需继续反向传播

      # 创建与当前层向量同形的零向量
      vector_1 = np.zeros_like(vectors[k])

      # 在目标神经元位置填充前一层对应神经元的激活值
      # 这对应于权重梯度计算中的前向激活值项
      vector_1[i] = vectors[l][j]

      # 计算梯度：局部梯度与输入激活值的乘积
      output = vector_2 * vector_1

   # 多层传播情况：需要通过递归传播误差信号
   else:  
      # 递归计算深层梯度，通过矩阵乘法@传播误差信号
      # 符合链式法则的神经网络反向传播特性
      output = vector_2 * (
         weights[k_layer]  @ diff_vect_weight(
            vectors, 
            intermediate_vectors, 
            weights, 
            activation_weights, 
            diff_activation, 
            k,       # 保持目标层不变
            l + 1,   # 向权重层靠近一层（递归向下传播）
            i + 1,   # 调整神经元索引（转回1-based索引）
            j + 1    # 调整权重索引（转回1-based索引）
         )
      )

   return output