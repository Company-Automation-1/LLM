import numpy as np
import test
import test2
import test3

# a = np.array([
#     [1, 2, 3],
#     [5, 5, 5],
#     [8, 7, 6],
#     [8, 8, 8]
# ])
# b = np.array([
#   [1, 1, 1, 1],
#   [2, 1, 1, 1],
#   [3, 1, 1, 1]
# ])
c = np.array([1, 2, 3])
# d = np.array([
#   [0],
#   [1],
#   [0],
# ])
# e = np.array([
#   [1],
#   [1],
#   [1],
# ])

# print(test3.quadratic_loss_function(d, e))
# print(test3.diagonal_product(a, b))
test3.vandermonde_matrix(c,3,test3.power)