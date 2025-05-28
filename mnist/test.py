import os
import torch
from torchvision import transforms
from PIL import Image
from .model import CNNNet

def simple_preprocess(image_path):
    """
    仅resize到28x28并归一化，不做灰度化，假设图片已全部手动处理好。
    """
    img = Image.open(image_path)
    img = img.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = transform(img)
    return img

def batch_predict(img_dir, model_path="mnist_cnn.pth"):
    """
    批量对img_dir下所有图片进行预测
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 支持常见图片格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    img_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[0]))  # 按数字排序，支持前缀

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        image = simple_preprocess(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1, keepdim=True)
        print(f"{img_file} 预测结果: {pred.item()}")

if __name__ == "__main__":
    batch_predict("test_imgs")