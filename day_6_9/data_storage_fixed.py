import numpy as np
import os
import pickle
import datetime

def save_model(weights, activation_weights, biases, filepath=None):
    """
    保存神经网络模型参数到文件
    
    参数:
        weights: 权重矩阵列表
        activation_weights: 激活权重列表
        biases: 偏置向量列表
        filepath: 保存路径，默认为当前时间命名的文件
    
    返回:
        filepath: 实际保存的文件路径
    """
    # 创建一个包含所有模型参数的字典
    model_data = {
        'weights': [w.tolist() for w in weights],
        'activation_weights': [aw.tolist() for aw in activation_weights],
        'biases': [b.tolist() for b in biases],
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    # 如果没有指定文件路径，则创建一个默认路径
    if filepath is None:
        # 确保matrix_data目录存在
        os.makedirs('day_6_9/matrix_data', exist_ok=True)
        filepath = f'day_6_9/matrix_data/model_{model_data["timestamp"]}.pkl'
    
    # 保存模型数据
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"模型已保存到: {filepath}")
    return filepath

def load_model(filepath):
    """
    从文件加载神经网络模型参数
    
    参数:
        filepath: 模型文件路径
    
    返回:
        weights, activation_weights, biases: 模型参数
    """
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 将列表转换回numpy数组
        weights = [np.array(w) for w in model_data['weights']]
        activation_weights = [np.array(aw) for aw in model_data['activation_weights']]
        biases = [np.array(b) for b in model_data['biases']]
        
        print(f"模型已从 {filepath} 加载")
        return weights, activation_weights, biases
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None, None

def save_training_data(vectors, intermediate_vectors, training_labels=None, filepath=None):
    """
    保存训练数据
    
    参数:
        vectors: 向量列表
        intermediate_vectors: 中间向量列表
        training_labels: 训练标签(可选)
        filepath: 保存路径，默认为当前时间命名的文件
    
    返回:
        filepath: 实际保存的文件路径
    """
    # 创建一个包含所有训练数据的字典
    training_data = {
        'vectors': [v.tolist() for v in vectors],
        'intermediate_vectors': [iv.tolist() for iv in intermediate_vectors],
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    if training_labels is not None:
        training_data['labels'] = training_labels.tolist() if isinstance(training_labels, np.ndarray) else training_labels
    
    # 如果没有指定文件路径，则创建一个默认路径
    if filepath is None:
        # 确保matrix_data目录存在
        os.makedirs('day_6_9/matrix_data', exist_ok=True)
        filepath = f'day_6_9/matrix_data/training_data_{training_data["timestamp"]}.pkl'
    
    # 保存训练数据
    with open(filepath, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"训练数据已保存到: {filepath}")
    return filepath

def load_training_data(filepath):
    """
    从文件加载训练数据
    
    参数:
        filepath: 训练数据文件路径
    
    返回:
        vectors, intermediate_vectors, training_labels: 训练数据
    """
    try:
        with open(filepath, 'rb') as f:
            training_data = pickle.load(f)
        
        # 将列表转换回numpy数组
        vectors = [np.array(v) for v in training_data['vectors']]
        intermediate_vectors = [np.array(iv) for iv in training_data['intermediate_vectors']]
        
        training_labels = None
        if 'labels' in training_data:
            training_labels = np.array(training_data['labels']) if isinstance(training_data['labels'], list) else training_data['labels']
        
        print(f"训练数据已从 {filepath} 加载")
        return vectors, intermediate_vectors, training_labels
    except Exception as e:
        print(f"加载训练数据时出错: {e}")
        return None, None, None

def list_saved_files(directory='day_6_9/matrix_data'):
    """
    列出保存的所有模型和训练数据文件
    
    参数:
        directory: 目录路径
    
    返回:
        files_dict: 包含模型文件和训练数据文件的字典
    """
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)
    
    # 获取目录中的所有文件
    all_files = os.listdir(directory)
    
    # 分类文件
    model_files = [f for f in all_files if f.startswith('model_')]
    training_files = [f for f in all_files if f.startswith('training_data_')]
    
    files_dict = {
        'model_files': [os.path.join(directory, f) for f in model_files],
        'training_files': [os.path.join(directory, f) for f in training_files]
    }
    
    return files_dict