import torch

#TODO: Softplus激活函数
def softplus(value, dummy):
    """
    Softplus激活函数
    
    实现了带有幂运算的softplus激活函数：log(1 + e^x)^p
    
    参数:
    - vector: 输入向量
    - p: 幂次参数，默认为1
    
    返回:
    - softplus激活后的向量
    """
    return torch.log1p(torch.exp(value)) * dummy

#TODO: Sigmoid激活函数
def sigmoid(value, dummy):
    """
    Sigmoid激活函数
    
    实现了带有幂运算的sigmoid激活函数：(1 / (1 + e^-x))^p
    
    参数:
    - vector: 输入向量
    - p: 幂次参数，默认为1
    
    返回:
    - sigmoid激活后的向量
    """
    return 1 / (1 + torch.exp(-value)) * dummy