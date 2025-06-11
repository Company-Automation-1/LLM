import numpy as np

#TODO: softplus 激活函数
def softplus(value, dummy):
    return np.log(1 + np.exp(value)) * dummy

#TODO: sigmoid 激活函数
def sigmoid(value, dummy):
    """
    softplus 的微分
    """
    return 1 / (1 + np.exp(-value)) * dummy
