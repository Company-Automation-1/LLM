# 《深度学习入门-基于Python的理论与实现》

## 项目结构

项目已按功能模块重新组织，目录结构如下：

```
src/
├── activations/              # 激活函数相关模块
│   ├── activation_functions.py  # 包含softplus和sigmoid等激活函数
│   └── __init__.py
├── gui/                      # 图形界面相关模块
│   └── main.py               # GUI主程序
├── models/                   # 模型保存和加载相关模块
│   ├── model_io.py           # 模型IO操作函数
│   ├── model_test.py         # 模型测试函数
│   └── __init__.py
├── networks/                 # 神经网络核心功能模块
│   ├── core.py               # 网络核心函数
│   └── __init__.py
├── trainers/                 # 训练器相关模块
│   ├── backprop.py           # 反向传播实现
│   └── __init__.py
├── utils/                    # 通用工具函数
│   ├── print_utils.py        # 打印工具函数
│   └── __init__.py
├── run_test.py               # 测试运行脚本
└── __init__.py
```

## 模块说明

- **activations**: 激活函数模块，包含各种激活函数的实现
- **gui**: 图形界面模块，包含交互界面相关代码
- **models**: 模型相关模块，包含模型保存、加载和管理相关功能
- **networks**: 神经网络核心模块，包含网络的基本组件和操作
- **trainers**: 训练器模块，包含反向传播和梯度计算相关功能
- **utils**: 通用工具模块，包含打印和调试等辅助功能

## 运行方式

测试反向传播算法：
```bash
python src/run_test.py
```

测试模型保存和加载：
```bash
python src/models/model_test.py
```

## 技术栈

- NumPy: 用于矩阵运算和神经网络数值计算
- PyQt5: 用于图形界面展示
- PIL/Pillow: 用于图像处理 