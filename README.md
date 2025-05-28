# MNIST手写数字识别项目

## 项目结构

```
.
├── README.md              # 项目说明
├── requirements.txt       # 依赖包
├── main.py                # 统一入口
├── mnist/
│   ├── __init__.py
│   ├── data.py            # 数据加载与预处理
│   ├── model.py           # 模型结构
│   ├── train.py           # 训练与验证逻辑
│   └── predict.py         # 预测逻辑
```

## 依赖环境
- Python 3.8+
- torch==2.6.0+cu124
- torchvision
- pillow

## 安装依赖
```bash
pip install -r requirements.txt
```

## 训练模型
```bash
python main.py train
```

## 预测图片
```bash
python main.py predict <图片路径>
```

## 说明
- 训练完成后会生成 `mnist_model.pth`。
- 预测时图片会自动转换为28x28灰度图。
- 支持GPU自动切换。 