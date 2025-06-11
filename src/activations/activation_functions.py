import numpy as np

def softplus(value, dummy):
    """
    Softplus激活函数: ln(1 + e^x)
    
    参数:
    - value: 输入值
    - dummy: 控制输出比例的参数
    
    返回:
    - softplus激活后的值
    """
    return np.log(1 + np.exp(value)) * dummy

def sigmoid(value, dummy):
    """
    Sigmoid激活函数: 1/(1 + e^(-x))
    实际上是softplus的微分
    
    参数:
    - value: 输入值
    - dummy: 控制输出比例的参数
    
    返回:
    - sigmoid激活后的值
    """
    return 1 / (1 + np.exp(-value)) * dummy 