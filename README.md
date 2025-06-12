# 《深度学习入门-基于Python的理论与实现》

## 项目结构

```
LLM/
├── activations.py           # 激活函数实现，如softplus和sigmoid等
├── main.py                  # 主程序入口和GUI实现
├── model.py                 # 模型保存和加载功能
├── networks.py              # 神经网络核心功能实现
├── run_test.py              # 测试运行脚本
├── trainers.py              # 训练相关功能，包含反向传播和梯度计算
├── util.py                  # 通用工具函数
├── qt/                      # Qt图形界面相关组件
│   ├── layout.py            # 界面布局
│   ├── layout.ui            # 界面设计文件
│   └── paintboard.py        # 画板实现
├── mnist/                   # MNIST参考项目
└── dataset/                 # MNIST数据集
```

## 模块说明

- **activations.py**: 激活函数模块，包含softplus和sigmoid等激活函数的实现
- **main.py**: 主程序入口，包含GUI界面实现和神经网络应用
- **model.py**: 模型管理模块，包含模型保存、加载和信息获取功能
- **networks.py**: 神经网络核心模块，包含网络的基本组件和前向传播
- **trainers.py**: 训练器模块，包含反向传播和梯度计算相关功能
- **util.py**: 通用工具模块，包含辅助功能
- **qt/**: Qt图形界面组件，包含界面布局和画板实现

## 运行方式

启动主程序（包含GUI界面）：
```bash
python main.py
```

运行测试：
```bash
python run_test.py
```

## 技术栈

- NumPy: 用于矩阵运算和神经网络数值计算
- PyQt5: 用于图形界面展示
- PIL/Pillow: 用于图像处理 