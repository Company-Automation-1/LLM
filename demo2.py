import numpy as np

def activation_function(x):
    """激活函数（示例使用ReLU）"""
    # 对应公式中的 σ(⋅) 操作
    # 数学公式: σ(linear_part) = max(0, linear_part)
    return np.maximum(0, x)

def amoeba_forward(x, W_k, W_k_sigma, beta_k, b_k):
    """
    Amoeba神经网络层的前向传播
    数学公式:
    x_{k+1} = (W_k_sigma ⋅ σ(W_k x_k + b_k) + β_k) ⊙ (W_k x_k + b_k)
    参数:
    x         : 输入向量 (形状: d_k × 1)
    W_k       : 权重矩阵 (形状: d_{k+1} × d_k)
    W_k_sigma : 激活路径权重矩阵 (形状: d_{k+1} × d_{k+1})
    beta_k    : 激活路径偏置 (形状: d_{k+1} × 1)
    b_k       : 线性偏置 (形状: d_{k+1} × 1)
    
    返回:
    x_{k+1}  : 输出向量 (形状: d_{k+1} × 1)
    """
    # 线性变换: W_k x_k + b_k
    # 对应公式: linear_part = W_k x_k + b_k ∈ R^{d_{k+1}×1}
    linear_part = np.dot(W_k, x) + b_k  # 形状: (d_{k+1} × 1)
    
    # 激活路径计算
    # 1. 激活函数: σ(linear_part)
    sigma_part = activation_function(linear_part)  # σ(W_k x_k + b_k)
    
    # 2. 激活路径线性变换: W_k_sigma ⋅ σ(linear_part) + beta_k
    # 对应公式: activation_path = W_k_sigma ⋅ σ(linear_part) + β_k ∈ R^{d_{k+1}×1}
    activation_path = np.dot(W_k_sigma, sigma_part) + beta_k
    
    # 哈达玛积（逐元素相乘）
    # 对应公式: x_{k+1} = activation_path ⊙ linear_part
    x_next = np.multiply(activation_path, linear_part)
    
    return x_next

# 示例运行
if __name__ == "__main__":
    # 输入向量 x_k (d_k = 3)
    x = np.array([[1.0], [2.0], [3.0]])  # 形状: 3×1 (即 x_k ∈ R^{3×1})

    # 权重矩阵 W_k (d_{k+1}=2 × d_k=3)
    W_k = np.random.randn(2, 3)          # W_k ∈ R^{2×3}

    # 激活路径权重矩阵 W_k_sigma (必须为 d_{k+1}×d_{k+1} = 2×2)
    W_k_sigma = np.random.randn(2, 2)    # W_k_sigma ∈ R^{2×2}

    # 偏置向量 b_k (2×1)
    b_k = np.random.randn(2, 1)          # b_k ∈ R^{2×1}

    # 激活路径偏置 beta_k (2×1)
    beta_k = np.random.randn(2, 1)       # β_k ∈ R^{2×1}

    # 前向传播计算
    x_next = amoeba_forward(x, W_k, W_k_sigma, beta_k, b_k)
    print("输出 x_{k+1}:\n", x_next)      # 结果形状: 2×1 (x_{k+1} ∈ R^{2×1})