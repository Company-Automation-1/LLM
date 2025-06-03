import numpy as np

def vector_power(vector, max_power):
    vector = np.asarray(vector).reshape(-1, 1)
    powers = np.arange(1, max_power + 1)
    return (vector ** powers).T

def propagate(vector, weights, activation_weight, biases):
    max_power = activation_weight.shape[1]
    vandermonde_matrix = vector_power(weights @ vector + biases, max_power)
    
    output_vector = np.array([])
    for i in range(activation_weight.shape[0]):
        output_vector = np.append(output_vector, activation_weight[i, :] @ vandermonde_matrix[:, i]).reshape(-1, 1)
    
    return output_vector

def softmax(x, temperature=1.0):
    max_x = np.max(x)
    y = np.exp((x - max_x) / temperature)
    f_x = y / np.sum(y)
    return f_x

def neural_network(input_vector, weights, activation_weights, biases): 
    vector = propagate(input_vector, weights, activation_weights, biases)
    output_vector = softmax(vector)
    return output_vector 