import numpy as np
import os
import pickle
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid

def diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k, l, i, j):
    
    k = k - 1
    l = l - 1
    i = i - 1
    j = j - 1


    degree = activation_weights[l].shape[1] - 1

    diff_activation_vector = vandermonde_matrix(intermediate_vectors[l], diff_activation, degree)

    print(f"diff_activation_vector:\n{diff_activation_vector}\n")

    vector_2 = diagonal_product(activation_weights[l], diff_activation_vector)

    if k == l + 1:
        vector_1 = np.zeros_like(vectors[k].T)

        vector_1[i] = vectors[l][j]


        output = diagonal_product(vector_2, vector_1)
    
    else:
        
        output = vector_2 * (weights[l] @ diff_vect_weight(vectors, intermediate_vectors, weights, activation_weights, diff_activation, k - 1, l, i, j))

    return output

# 添加模型保存功能
def save_model(model_data, filepath, format='pickle'):
    """
    保存神经网络模型参数到文件
    
    参数:
        model_data: 包含模型参数的字典，可以包含:
            - vectors: 向量列表
            - intermediate_vectors: 中间向量列表
            - weights: 权重矩阵列表
            - activation_weights: 激活权重列表
            - biases: 偏置向量列表(可选)
            - 其他自定义参数
        filepath: 文件保存路径
        format: 保存格式，支持'pickle'或'numpy'
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    elif format == 'numpy':
        np.savez(filepath, **model_data)
    else:
        raise ValueError(f"不支持的格式: {format}，请使用 'pickle' 或 'numpy'")
    
    print(f"模型已保存到: {filepath}")

# 添加模型加载功能
def load_model(filepath, format='pickle'):
    """
    从文件加载神经网络模型参数
    
    参数:
        filepath: 文件加载路径
        format: 保存格式，支持'pickle'或'numpy'
    
    返回:
        包含模型参数的字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到模型文件: {filepath}")
    
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    elif format == 'numpy':
        loaded_data = np.load(filepath, allow_pickle=True)
        model_data = {key: loaded_data[key] for key in loaded_data.files}
    else:
        raise ValueError(f"不支持的格式: {format}，请使用 'pickle' 或 'numpy'")
    
    print(f"模型已从 {filepath} 加载")
    return model_data

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
    
    # 示例：如何保存模型
    model_data = {
        'vectors': vectors,
        'intermediate_vectors': intermediate_vectors,
        'weights': weights,
        'activation_weights': activation_weights,
        'activation_function': 'sigmoid'  # 记录使用的激活函数名称
    }
    
    # 确保models目录存在
    os.makedirs("models", exist_ok=True)
    
    # 保存模型示例
    save_model(model_data, "models/trainer_model.pkl")
    
    # 加载模型示例
    loaded_model = load_model("models/trainer_model.pkl")
    
    # 验证加载的模型
    print("\n验证加载的模型:")
    print(f"原始weights形状: {weights[0].shape}")
    print(f"加载的weights形状: {loaded_model['weights'][0].shape}")
    print(f"weights一致性检查: {np.array_equal(weights[0], loaded_model['weights'][0])}")
