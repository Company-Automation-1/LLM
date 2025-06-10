# 幂次激活神经网络模型 - 持久化功能

本项目实现了一种基于幂次激活函数的神经网络模型，并提供了模型持久化（保存和加载）功能。

## 文件结构

- `main_Lib_1.py`: 核心算法库，包含神经网络前向传播函数
- `main_Lib_2.py`: 辅助激活函数库
- `trainer.py`: 训练器实现，现已添加模型保存和加载功能
- `model_saver.py`: 独立的模型保存和加载的核心功能
- `model_trainer.py`: 封装的神经网络训练器类
- `trainer_extension.py`: 扩展trainer.py的功能，添加保存/加载支持

## 如何使用trainer.py中的持久化功能

### 保存模型

```python
from trainer import save_model

# 准备模型数据
model_data = {
    'vectors': vectors,                   # 向量列表
    'intermediate_vectors': intermediate_vectors,  # 中间向量列表
    'weights': weights,                   # 权重矩阵列表
    'activation_weights': activation_weights,     # 激活权重列表
    'activation_function': 'sigmoid'      # 记录使用的激活函数名称(可选)
}

# 保存模型
save_model(model_data, "models/my_model.pkl")

# 如果要使用numpy格式保存
save_model(model_data, "models/my_model.npz", format='numpy')
```

### 加载模型

```python
from trainer import load_model
from main_Lib_2 import sigmoid  # 加载您使用的激活函数

# 加载模型
loaded_model = load_model("models/my_model.pkl")

# 从加载的模型中提取参数
vectors = loaded_model['vectors']
intermediate_vectors = loaded_model['intermediate_vectors']
weights = loaded_model['weights']
activation_weights = loaded_model['activation_weights']

# 使用加载的模型进行计算
from trainer import diff_vect_weight

result = diff_vect_weight(
    vectors, 
    intermediate_vectors, 
    weights, 
    activation_weights, 
    sigmoid,  # 使用适当的激活函数
    2, 1, 1, 1  # 其他参数
)
```

## 支持的保存格式

- `pickle`: 默认格式，使用Python的pickle模块
- `numpy`: 使用numpy的npz格式

## 注意事项

1. 保存模型时，请确保包含神经网络计算所需的所有参数
2. 激活函数（如sigmoid）不会直接保存，需要在加载后重新指定
3. 对于大型模型，推荐使用numpy格式保存以获得更好的兼容性 