import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    """
    卷积神经网络（CNN）用于手写数字识别。
    结构：
    - 2层卷积+ReLU+池化
    - Dropout防止过拟合
    - 2层全连接
    """
    def __init__(self):
        super(CNNNet, self).__init__()
        # 第一层卷积：输入通道1，输出通道32，卷积核3x3，步长1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二层卷积：输入通道32，输出通道64，卷积核3x3，步长1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout层，防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 第一个全连接层，输入9216（64*12*12），输出128
        self.fc1 = nn.Linear(9216, 128)
        # 第二个全连接层，输出10类（数字0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播过程：输入x，输出分类结果
        """
        # 第一层卷积+ReLU激活
        x = self.conv1(x)
        x = F.relu(x)
        # 第二层卷积+ReLU激活
        x = self.conv2(x)
        x = F.relu(x)
        # 2x2最大池化，降维
        x = F.max_pool2d(x, 2)
        # Dropout，防止过拟合
        x = self.dropout1(x)
        # 展平成一维向量
        x = torch.flatten(x, 1)
        # 全连接层+ReLU
        x = self.fc1(x)
        x = F.relu(x)
        # Dropout
        x = self.dropout2(x)
        # 输出层（未激活，交叉熵损失自带softmax）
        x = self.fc2(x)
        return x 