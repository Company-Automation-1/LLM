import numpy as np
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid

def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):

    vector_2 = diagonal_product(activation_weights[l], diff_activation_vector)

    if k == l + 1:
      vector_1 = np.zeros_like(vectors[k].T)

      vector_1[i] = vectors[l][j]

      degree = activation_weights.shape[1]

      diff_activation_vector = vandermonde_matrix(intermediate_vectors[l], diff_activation, degree)

      output = diagonal_product(vector_2, vector_1)
    
    else:
      output = vector_2 * (weights[l] @ diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k - 1, l, i, j))

    return output


if __name__ == '__main__':
    vector_1 = np.array([1, 2, 3, 4]).T
    vector_2 = np.array([5, 6, 7, 8]).T

    vectors = [vector_1, vector_2]
    print(vectors)