import numpy as np
import os
import pickle

def save_model(model_params, filepath, format='pickle'):
    """
    保存神经网络模型参数到文件
    
    参数:
        model_params: 包含模型参数的字典，例如:
            - weights: 权重矩阵列表
            - activation_weights: 激活权重列表
            - biases: 偏置向量列表
            - temperature: softmax温度参数(可选)
        filepath: 文件保存路径
        format: 保存格式，支持'pickle'或'numpy'
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model_params, f)
    elif format == 'numpy':
        np.savez(filepath, **model_params)
    else:
        raise ValueError(f"不支持的格式: {format}，请使用 'pickle' 或 'numpy'")
    
    print(f"模型已保存到: {filepath}")

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
            model_params = pickle.load(f)
    elif format == 'numpy':
        model_data = np.load(filepath, allow_pickle=True)
        model_params = {key: model_data[key] for key in model_data.files}
    else:
        raise ValueError(f"不支持的格式: {format}，请使用 'pickle' 或 'numpy'")
    
    print(f"模型已从 {filepath} 加载")
    return model_params

    """
    获取模型结构信息的简要描述
    
    参数:
        model_params: 模型参数字典
    
    返回:
        包含模型结构信息的字符串
    """
    weights = model_params['weights']
    activation_weights = model_params['activation_weights']
    biases = model_params['biases']
    
    info = []
    info.append(f"网络层数: {len(weights)}")
    
    for i, (w, aw, b) in enumerate(zip(weights, activation_weights, biases)):
        info.append(f"第{i+1}层:")
        info.append(f"  输入维度: {w.shape[1]}")
        info.append(f"  输出维度: {w.shape[0]}")
        info.append(f"  激活维度: {aw.shape[1]}")
    
    return "\n".join(info) 