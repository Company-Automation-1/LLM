import torch
import torch.optim as optim
import torch.nn as nn
from .model import CNNNet
from .data import get_mnist_loaders

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    单轮训练过程。
    model: 神经网络模型
    device: 运行设备（CPU/GPU）
    train_loader: 训练集DataLoader
    optimizer: 优化器
    criterion: 损失函数
    epoch: 当前轮数
    """
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')

def run():
    """
    训练标准MNIST数据集。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders()
    model = CNNNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, criterion, epoch)
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("模型已保存为 mnist_cnn.pth") 