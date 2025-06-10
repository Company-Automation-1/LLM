import numpy as np
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid

def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):

   k = k - 1
   l = l - 1
   i = i - 1
   j = j - 1

   degree = activation_weights[k - 1].shape[1] - 1

   diff_activation_vector = vandermonde_matrix(intermediate_vectors[k - 1], diff_activation, degree)

   vector_2 = diagonal_product(activation_weights[k - 1], diff_activation_vector)

   if k == l + 1:

      vector_1 = np.zeros_like(vectors[k])

      vector_1[i] = vectors[l][j]

      output = vector_2 * vector_1

   else:
      b = diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l + 1, i + 1, j + 1)

      a = weights[k - 1]

      c = a @ b

      output = vector_2 * c

   return output

if __name__ == '__main__':
   vector_1 = np.array([
      [1, 1, 1, 1, 1]
   ]).T
   vector_2 = np.array([
      [1, 2, 3, 4]
   ]).T
   vector_3 = np.array([
      [4, 5, 6]
   ]).T
   
   intermediate_vector_1 = np.array([
      [5, 5, 5, 5, 5]
   ]).T
   intermediate_vector_2 = np.array([
      [5, 5, 5, 5]
   ]).T

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

   vectors = [vector_3, vector_2, vector_1]
   intermediate_vectors = [intermediate_vector_2, intermediate_vector_1]
   weights = [weight_2, weight_1]
   activation_weights = [activation_weight_2, activation_weight_1]

   result = diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, softplus, 3, 1, 1, 2)

   print(result)
