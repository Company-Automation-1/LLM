import torch
from torchvision import transforms
from PIL import Image
from .model import CNNNet

def predict(image_path, model_path="mnist_cnn.pth"):
    """
    加载模型并对单张图片进行预测。
    image_path: 图片路径
    model_path: 模型文件路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNet().to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 定义与训练一致的预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载并预处理图片
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    print(f"预测结果: {pred.item()+1}")  # 标签从0开始，+1对应原始图片名
    return pred.item()+1 