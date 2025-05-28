import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDigitDataset(Dataset):
    """
    用于加载自定义图片（如img目录下的1.jpg~9.jpg），并自动生成标签。
    每张图片的文件名为数字.jpg，标签自动从文件名提取。
    """
    def __init__(self, img_dir, transform=None):
        """
        img_dir: 图片文件夹路径
        transform: 图像预处理方法
        """
        self.img_dir = img_dir
        # 只读取.jpg文件，并按数字排序
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        self.transform = transform
        # 标签直接从文件名提取（如1.jpg->1）
        self.labels = [int(os.path.splitext(f)[0]) for f in self.img_files]

    def __len__(self):
        # 返回图片数量
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图片并转为灰度
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('L')
        # 标签从0开始
        label = self.labels[idx] - 1
        # 应用预处理
        if self.transform:
            image = self.transform(image)
        return image, label 