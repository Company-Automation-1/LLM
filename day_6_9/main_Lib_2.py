import numpy as np

#TODO: softplus 激活函数
def softplus(value):
    return np.log(1 + np.exp(value))

#TODO: sigmoid 激活函数
def sigmoid(value):
    """
    softplus 的微分
    """
    return 1 / (1 + np.exp(-value))
