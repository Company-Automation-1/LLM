import numpy as np

# Vandermonde矩陣
def vector_power(vector, max_power):
    vector = np.asarray(vector).reshape(-1, 1)
    powers = np.arange(1, max_power + 1)
    return (vector ** powers).T

# Propagate函數
def propagate(vector, weights, activation_weight, biases):
    max_power = activation_weight.shape[1]
    print(f"max_power: {max_power}")
    print(f"vector:\n{vector}\nweights:\n{weights}\nactivation_weight:\n{activation_weight}\nbiases:\n{biases}")
    print(weights @ vector + biases)
    vandermonde_matrix = vector_power(weights @ vector + biases, max_power)
    
    output_vector = np.array([])
    for i in range(activation_weight.shape[0]):
        output_vector = np.append(output_vector, activation_weight[i, :] @ vandermonde_matrix[:, i]).reshape(-1, 1)
     
    return output_vector

# Softmax函數
def softmax(x, temperature=1):
    max_x = np.max(x)
    y = np.exp((x - max_x) / temperature)
    f_x = y / np.sum(y)
    return f_x

# 神經網絡
def neural_network(input_vector, weights, activation_weights, biases): 
    vector = propagate(input_vector, weights, activation_weights, biases)
    output_vector = softmax(vector)
    return output_vector


def test_neural_network():
    """
    测试神经网络前向传播
    """
    # 1. 定义测试数据
    input_vector = np.array([[1, 2, 3]]).T
    weights = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1], [4, 4, 4]])
    activation_weights = np.array([[4, 3, 5], [2, 1, 2], [1, 2, 6], [3, 2, 4]])
    biases = np.array([[2, 2, 1, 3]]).T

    # 2. 调用前向传播函数
    output_vector = neural_network(input_vector, weights, activation_weights, biases)

    # 3. 打印输出结果
    # print(output_vector)
    
    # [[0.]
    #  [0.]
    #  [0.]
    #  [1.]]

# 执行测试
if __name__ == "__main__":
    test_neural_network()