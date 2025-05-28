import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, num_workers=2):
    """
    获取MNIST数据集的训练和测试DataLoader。
    参数：
        batch_size: 每个batch的图片数量
        num_workers: 加载数据的线程数
    返回：
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
    """
    # 定义数据预处理：转为Tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 下载并加载训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 下载并加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader 