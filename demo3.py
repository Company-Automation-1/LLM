import numpy as np
a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

b = np.array([[10, 11, 12],[13, 14, 15],[16, 17, 18]])

c = np.stack((a, b), axis=0)

# print(c)

def matrix(a, b):
    return np.stack((a, b), axis=0)


# def neural_network(input_vector, weights, activation_weights, biases):
#     layer_1 = np.dot(input_vector, weights) + biases

def softmax(x):

    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def test_softmax():
    x = np.array([1, 2, 3, 4])
    print(softmax(x))

def test_matrix():
    matrix_A=np.array([[1,2]])
    matrix_B=np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(matrix(matrix_A, matrix_B))

if __name__ == "__main__":
    # test_softmax()
    test_matrix()