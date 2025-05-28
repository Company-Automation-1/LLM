import sys

def main():
    """
    项目主入口。支持两种模式：
    - train: 训练标准MNIST
    - predict: 对单张图片进行预测
    """
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "predict"]:
        print("用法: python main.py [train|predict] [图片路径] [模型文件]")
        return
    mode = sys.argv[1]
    if mode == "train":
        # 标准MNIST训练
        from mnist.train import run
        run()
    elif mode == "predict":
        # 单张图片预测
        if len(sys.argv) < 3:
            print("用法: python main.py predict <图片路径> [模型文件]")
            return
        from mnist.predict import predict
        model_path = sys.argv[3] if len(sys.argv) > 3 else "mnist_cnn.pth"
        predict(sys.argv[2], model_path)

if __name__ == "__main__":
    main()
