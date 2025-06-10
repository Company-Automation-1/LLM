import numpy as np
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
from main_Lib_2 import softplus, sigmoid
from data_storage_fixed import save_model, load_model, save_training_data, load_training_data, list_saved_files

def demo_save_and_load():
    """
    演示如何保存和加载模型参数与训练数据
    """
    print("=== 数据存储模块演示 ===")
    
    # 创建示例数据
    print("\n1. 创建示例数据...")
    
    # 创建权重、激活权重和偏置
    weights = [
        np.array([
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5]
        ])
    ]
    
    activation_weights = [
        np.array([
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]
        ])
    ]
    
    biases = [
        np.array([
            [1],
            [1],
            [1],
            [1]
        ])
    ]
    
    # 创建向量和中间向量
    vectors = [
        np.array([[1, 2, 3, 4]]).T,
        np.array([[4, 5, 6]]).T
    ]
    
    intermediate_vectors = [
        np.array([[5, 5, 5, 5]]).T
    ]
    
    # 示例标签
    labels = np.array([0, 1, 2])
    
    # 保存模型
    print("\n2. 保存模型参数...")
    model_file = save_model(weights, activation_weights, biases)
    
    # 保存训练数据
    print("\n3. 保存训练数据...")
    data_file = save_training_data(vectors, intermediate_vectors, labels)
    
    # 列出已保存的文件
    print("\n4. 列出已保存的文件...")
    files = list_saved_files()
    print(f"模型文件: {files['model_files']}")
    print(f"训练数据文件: {files['training_files']}")
    
    # 加载模型
    print("\n5. 加载模型参数...")
    loaded_weights, loaded_activation_weights, loaded_biases = load_model(model_file)
    
    # 验证加载的模型参数
    print("\n6. 验证加载的模型参数...")
    print(f"原始权重形状: {weights[0].shape}, 加载后权重形状: {loaded_weights[0].shape}")
    print(f"权重数据是否一致: {np.array_equal(weights[0], loaded_weights[0])}")
    
    # 加载训练数据
    print("\n7. 加载训练数据...")
    loaded_vectors, loaded_intermediate_vectors, loaded_labels = load_training_data(data_file)
    
    # 验证加载的训练数据
    print("\n8. 验证加载的训练数据...")
    print(f"原始向量形状: {vectors[0].shape}, 加载后向量形状: {loaded_vectors[0].shape}")
    print(f"向量数据是否一致: {np.array_equal(vectors[0], loaded_vectors[0])}")
    print(f"标签是否一致: {np.array_equal(labels, loaded_labels)}")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    demo_save_and_load()
