import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.networks.core import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network, diff_softmax
from src.activations.activation_functions import softplus, sigmoid
from src.utils.print_utils import pt
from src.trainers.backprop import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient
from src.models.model_io import load_model, save_model

def demo_model_save():
    """
    模型保存示例函数
    """
    # 示例模型数据
    model_data = {
        'weights': [np.random.randn(10, 784), np.random.randn(10, 10)],
        'activation_weights': [np.random.randn(10, 2), np.random.randn(10, 2)],
        'biases': [np.random.randn(10, 1), np.random.randn(10, 1)],
        'temperature': 1.0
    }

    # 确保models目录存在
    os.makedirs("models", exist_ok=True)

    # 保存模型示例
    save_model(model_data, "models/demo_model.pkl")
    
    return model_data

def demo_model_load():
    """
    模型加载示例函数
    """
    # 加载模型示例
    loaded_model = load_model("models/demo_model.pkl")

    # 打印模型信息
    pt('模型包含的键:', list(loaded_model.keys()))
    pt('模型信息示例:', {k: type(v) for k, v in loaded_model.items()})
    
    return loaded_model

def test_diff_softmax():
    """
    测试softmax导数计算
    """
    vector = np.array([[1, 2, 3, 4]]).T
    diff_vector = np.array([[1, 1, 1, 2]]).T

    result = diff_softmax(vector, diff_vector, 1)
    pt('diff_softmax测试结果', result)
    
    return result

if __name__ == "__main__":
    # 保存一个示例模型
    model_data = demo_model_save()
    
    # 加载刚才保存的模型
    loaded_model = demo_model_load()
    
    # 测试softmax导数
    test_diff_softmax() 