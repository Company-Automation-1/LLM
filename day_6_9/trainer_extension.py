import numpy as np
from trainer import diff_vect_weight
from model_saver import save_model, load_model, create_model_params, get_model_info

def save_training_state(vectors, intermediate_vectors, weights, activation_weights, 
                        biases=None, temperature=1.0, filepath="models/training_state.pkl", 
                        format='pickle', extra_params=None):
    """
    保存训练状态到文件
    
    参数:
        vectors: 向量列表
        intermediate_vectors: 中间向量列表
        weights: 权重矩阵列表
        activation_weights: 激活权重列表
        biases: 偏置向量列表(可选)
        temperature: softmax温度参数(可选)
        filepath: 文件保存路径
        format: 保存格式，支持'pickle'或'numpy'
        extra_params: 额外的模型参数字典(可选)
    """
    # 创建模型参数字典
    model_params = {
        'vectors': vectors,
        'intermediate_vectors': intermediate_vectors,
        'weights': weights,
        'activation_weights': activation_weights,
    }
    
    # 添加可选参数
    if biases is not None:
        model_params['biases'] = biases
    
    model_params['temperature'] = temperature
    
    # 添加额外参数
    if extra_params is not None:
        model_params.update(extra_params)
    
    # 保存模型
    save_model(model_params, filepath, format)
    
    return model_params

def load_training_state(filepath, format='pickle'):
    """
    从文件加载训练状态
    
    参数:
        filepath: 文件路径
        format: 文件格式
    
    返回:
        包含训练状态的字典
    """
    return load_model(filepath, format)

def get_training_state_info(training_state):
    """
    获取训练状态的信息
    
    参数:
        training_state: 训练状态字典
    
    返回:
        包含训练状态信息的字符串
    """
    vectors = training_state.get('vectors', [])
    intermediate_vectors = training_state.get('intermediate_vectors', [])
    weights = training_state.get('weights', [])
    activation_weights = training_state.get('activation_weights', [])
    
    info = []
    info.append(f"网络层数: {len(weights)}")
    
    info.append("\n向量信息:")
    for i, v in enumerate(vectors):
        info.append(f"  向量{i+1}: 形状={v.shape}")
    
    info.append("\n中间向量信息:")
    for i, v in enumerate(intermediate_vectors):
        info.append(f"  中间向量{i+1}: 形状={v.shape}")
    
    info.append("\n权重信息:")
    for i, w in enumerate(weights):
        info.append(f"  权重矩阵{i+1}: 形状={w.shape}")
    
    info.append("\n激活权重信息:")
    for i, aw in enumerate(activation_weights):
        info.append(f"  激活权重{i+1}: 形状={aw.shape}")
    
    # 添加额外信息
    for key, value in training_state.items():
        if key not in ['vectors', 'intermediate_vectors', 'weights', 'activation_weights', 'biases', 'temperature']:
            info.append(f"\n{key}: {value}")
    
    return "\n".join(info)

# 用法示例
if __name__ == "__main__":
    # 导入原始的trainer模块示例
    from trainer import diff_vect_weight
    from main_Lib_1 import power
    from main_Lib_2 import sigmoid
    
    # 创建一些示例数据（与trainer.py的示例类似）
    vector_1 = np.array([[1, 2, 3, 4]]).T
    vector_2 = np.array([[4, 5, 6]]).T
    intermediate_vector_1 = np.array([[5, 5, 5, 5]]).T
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
    biases = [np.zeros((4, 1))]
    
    vectors = [vector_1, vector_2]
    intermediate_vectors = [intermediate_vector_1]
    weights = [weight_1]
    activation_weights = [activation_weight_1]
    
    # 保存训练状态
    print("保存训练状态...")
    training_state = save_training_state(
        vectors=vectors,
        intermediate_vectors=intermediate_vectors,
        weights=weights,
        activation_weights=activation_weights,
        biases=biases,
        filepath="models/training_example_state.pkl",
        extra_params={
            "description": "示例训练状态",
            "activation_function": "sigmoid"
        }
    )
    
    # 输出训练状态信息
    print("\n训练状态信息:")
    print(get_training_state_info(training_state))
    
    # 加载训练状态
    print("\n加载训练状态...")
    loaded_state = load_training_state("models/training_example_state.pkl")
    
    # 验证加载的状态
    print("\n验证权重矩阵相同:")
    print(f"原始权重形状: {weights[0].shape}")
    print(f"加载权重形状: {loaded_state['weights'][0].shape}")
    print(f"权重相同: {np.array_equal(weights[0], loaded_state['weights'][0])}")
    
    # 使用加载的状态继续训练
    print("\n使用加载的状态调用原始diff_vect_weight函数:")
    result = diff_vect_weight(
        loaded_state['vectors'], 
        loaded_state['intermediate_vectors'], 
        loaded_state['weights'], 
        loaded_state['activation_weights'], 
        sigmoid, 2, 1, 1, 1
    )
    
    print(f"结果形状: {result.shape}") 