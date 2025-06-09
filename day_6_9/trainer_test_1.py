import numpy as np
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid

def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):
    
    k = k - 1
    l = l - 1
    i = i - 1
    j = j - 1

    diff_activation_vector = diff_activation(intermediate_vectors[l])

    vector_2 = diagonal_product(activation_weights[l], diff_activation_vector)

    print(vector_2)

    if k == l + 1:
        vector_1 = np.zeros_like(vectors[k].T)

        vector_1[i] = vectors[l][j]

        degree = activation_weights.shape[1]

        diff_activation_vector = vandermonde_matrix(intermediate_vectors[l], diff_activation, degree)

        output = diagonal_product(vector_2, vector_1)
    
    else:
        a = diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k - 1, l, i, j)
        output = vector_2 * (weights[l] @ a)

    return output

if __name__ == '__main__':
    vector_1 = np.array([
       [1, 2, 3, 4]
       ]).T
    vector_2 = np.array([
       [4, 5, 6]
       ]).T
    intermediate_vector_1 = np.array([
       [5, 5, 5, 5]
       ]).T
    weight_1 = np.array([
       [2, 2, 2],
       [3, 3, 3],
       [4, 4, 4],
       [5, 5, 5]
    ])
    activation_weight_1 = np.array([
       [1, 2],
       [1, 2],
       [1, 2],
       [1, 2]
    ])


    vectors = [vector_1, vector_2]
    intermediate_vectors = [intermediate_vector_1]
    weights = [weight_1]
    activation_weights = [activation_weight_1]


    result = diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, sigmoid, 2, 1, 1, 1)

    print(result)
