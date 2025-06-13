# 项目结构
项目包含两种实现方式：
- 根目录：使用NumPy实现的网络架构
- src目录：使用PyTorch实现的网络架构，与NumPy版本具有完全相同的数学模型，目的是利用GPU加速训练和推理

## 模型并行实现
- NumPy实现：用于概念验证和CPU上运行
- PyTorch实现：用于GPU加速，保持与NumPy版本相同的数学模型

## 使用方法
1. **启动程序**：运行`main.py`
2. **绘制数字**：在右侧画板上绘制数字（0-9）
3. **识别**：点击"识别"按钮进行预测
4. **训练**：
   - 在"正确结果"输入框中输入正确的数字
   - 设置学习率（默认0.1）
   - 点击"提交正确结果"按钮进行模型更新

## 依赖库
- NumPy：数值计算
- PyTorch：GPU加速
- PyQt5：图形界面
- PIL：图像处理

### 详细目录结构
```
app/
├── main.py                      # 主程序入口，包含GUI界面实现
├── activations.py               # NumPy版激活函数实现
├── networks.py                  # NumPy版网络层和前向传播实现
├── trainers.py                  # NumPy版反向传播和梯度计算实现
├── model.py                     # 模型保存和加载功能
├── util.py                      # 工具函数
├── README.md                    # 本文档
├── qt/                          # PyQt5 GUI相关模块
│   ├── layout.py                # 界面布局代码
│   ├── layout.ui                # 界面布局UI设计文件
│   └── paintboard.py            # 画板实现
└── src/                         # PyTorch实现版本
    ├── activations.py           # PyTorch版激活函数
    ├── networks.py              # PyTorch版网络层和前向传播
    ├── trainers.py              # PyTorch版反向传播和梯度计算
    └── train_single_sample_repeat.py  # 单样本重复训练脚本
```
