import torch
import copy
import time

from networks import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network, diff_softmax
from activations import softplus, sigmoid

#TODO: 权重梯度的反向传播
def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):

   # 转换为0-based索引
   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1

   k_layer = k - 1
   degree = activation_weights[k_layer].shape[1] - 1

   diff_activation_vector = vandermonde_matrix(
      intermediate_vectors[k_layer],
      diff_activation,
      degree
   )

   vector_2 = diagonal_product(
      activation_weights[k_layer],
      diff_activation_vector
   )

   if k == l + 1:
      vector_1 = torch.zeros_like(vectors[k])
      vector_1[i] = vectors[l][j]
      output = vector_2 * vector_1
   else:
      output = vector_2 * (
         torch.matmul(weights[k_layer], diff_vect_weight(
            vectors,
            intermediate_vectors,
            weights,
            activation_weights,
            diff_activation,
            k,
            l + 1,
            i + 1,
            j + 1
         ))
      )

   return output

#TODO: 激活权重梯度的反向传播
def diff_vect_aweight(vectors, intermediate_vectors, weights, activation_weights, activation_function, diff_activation, k, l, i, j):
        
   # 转换为0-based索引
   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1

   k_layer = k - 1

   if k == l + 1:
      vector_1 = torch.zeros_like(vectors[k])
      degree = activation_weights[k_layer].shape[1] - 1
      activated_vector = vandermonde_matrix(
         intermediate_vectors[k_layer],
         activation_function,
         degree
      )
      vector_1[i] = activated_vector[j][i]
      output = vector_1
   else:
      degree = activation_weights[k_layer].shape[1] - 1
      diff_activation_vector = vandermonde_matrix(
         intermediate_vectors[k_layer],
         diff_activation,
         degree
      )
      vector_2 = diagonal_product(
         activation_weights[k_layer],
         diff_activation_vector
      )
      output = vector_2 * (
         torch.matmul(weights[k_layer], diff_vect_aweight(
            vectors,
            intermediate_vectors,
            weights,
            activation_weights,
            activation_function,
            diff_activation,
            k,
            l + 1,
            i + 1,
            j + 1
         ))
      )

   return output

#TODO: 偏置梯度的反向传播
def diff_vect_bias(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, j):

   # 转换为0-based索引
   k = k - 1
   l = l - 1
   j = j - 1

   k_layer = k - 1
   degree = activation_weights[k_layer].shape[1] - 1

   diff_activation_vector = vandermonde_matrix(
      intermediate_vectors[k_layer],
      diff_activation,
      degree
   )

   vector_2 = diagonal_product(
      activation_weights[k_layer],
      diff_activation_vector
   )
   
   if k == l + 1:
      vector_1 = torch.zeros_like(vectors[k])
      vector_1[j] = vector_2[j]
      output = vector_1
   else:
      output = vector_2 * (
         torch.matmul(weights[k_layer], diff_vect_bias(
            vectors,
            intermediate_vectors,
            weights,
            activation_weights,
            diff_activation,
            k,
            l + 1,
            j + 1
         ))
      )

   return output

#TODO: 计算神经网络各参数的梯度
def gradient(vectors, 
            intermediate_vectors, 
            weights, 
            activation_weights, 
            biases, 
            activation_function, 
            diff_activation, 
            temperature, 
            actual):

  # 打印开始计算梯度
  # 绿色
  print(f"\033[92m开始计算梯度... 网络层数: {len(weights)}\033[0m")
  start_time = time.time()

  back_weights = copy.deepcopy(weights)
  back_act_weights = copy.deepcopy(activation_weights)
  back_biases = copy.deepcopy(biases)

  layers = len(weights)
  
  # 逐层计算梯度
  for layer in range(layers):
      
      layer_start_time = time.time()
      # 蓝色
      print(f"\033[94m计算第 {layer+1}/{layers} 层梯度...\033[0m")
      
      rows = weights[layer].shape[0]
      cols = weights[layer].shape[1]
      
      # 黄色
      print(f"\033[93m  计算权重(back_weights)梯度: 形状 {rows}x{cols}...\033[0m")
      
      weights_start_time = time.time()
      
      # 计算权重梯度
      for row in range(rows):

         if row % max(1, rows//5) == 0 or row == rows-1:  # 打印5次进度
            # 粉色
            print(f"\033[95m    权重(back_weights)行: {row+1}/{rows} 完成\033[0m")
          
         for col in range(cols):
            # 计算权重梯度
            back_weights[layer][row][col] = 2 * torch.matmul(
                (softmax(vectors[layers], temperature) - actual).t(),
                diff_softmax(
                    vectors[layers],
                    diff_vect_weight(
                        vectors,
                        intermediate_vectors,
                        weights,
                        activation_weights,
                        diff_activation,
                        len(vectors),
                        layer + 1,
                        row + 1,
                        col + 1
                    ),
                    temperature
                )
            )
 
      # 黄色
      print(f"\033[93m  权重梯度(back_weights)计算完成，耗时: {time.time() - weights_start_time:.2f}秒\033[0m")

      # 计算激活权重梯度
      rows = activation_weights[layer].shape[0]
      cols = activation_weights[layer].shape[1]
      
      # 黄色
      print(f"\033[93m  计算激活权重(back_act_weights)梯度: 形状 {rows}x{cols}...\033[0m")
      act_weights_start_time = time.time()
      for row in range(rows):
         if row % max(1, rows//5) == 0 or row == rows-1:  # 打印5次进度
            # 粉色
            print(f"\033[95m    激活权重(back_act_weights)行: {row+1}/{rows} 完成\033[0m")

         for col in range(cols):
            # 计算激活权重梯度
            back_act_weights[layer][row][col] = 2 * torch.matmul(
                (softmax(vectors[layers], temperature) - actual).t(),
                diff_softmax(
                    vectors[layers],
                    diff_vect_aweight(
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
                    ),
                    temperature
                )
            )
      
      # 黄色
      print(f"\033[93m  激活权重(back_act_weights)梯度计算完成，耗时: {time.time() - act_weights_start_time:.2f}秒\033[0m")
      
      # 计算偏置梯度
      rows = biases[layer].shape[0]
      # 黄色
      print(f"\033[93m  计算偏置(back_biases)梯度: 形状 {rows}x1...\033[0m")
      bias_start_time = time.time()
      for row in range(rows):
         if row % max(1, rows//5) == 0 or row == rows-1:  # 打印5次进度
            # 粉色
            print(f"\033[95m    偏置(back_biases)行: {row+1}/{rows} 完成\033[0m")

         # 计算偏置梯度
         back_biases[layer][row] = 2 * torch.matmul(
            (softmax(vectors[layers], temperature) - actual).t(),
            diff_softmax(
                vectors[layers],
                diff_vect_bias(
                    vectors,
                    intermediate_vectors,
                    weights,
                    activation_weights,
                    diff_activation,
                    len(vectors),
                    layer + 1,
                    row + 1
                ),
                temperature
            )
         )
      
      # 黄色
      print(f"\033[93m  偏置梯度(back_biases)计算完成，耗时: {time.time() - bias_start_time:.2f}秒\033[0m")

      # 蓝色
      print(f"\033[94m第 {layer+1}/{layers} 层梯度(weights)计算完成，总耗时: {time.time() - layer_start_time:.2f}秒\033[0m")

  total_time = time.time() - start_time

   # \033[92m 设置文本为绿色
  # \033[0m 重置颜色设置
  print(f"\033[92m梯度(weights)计算完成，总耗时: {total_time:.2f}秒\033[0m")
  
  # 返回所有梯度
  return back_weights, back_act_weights, back_biases 