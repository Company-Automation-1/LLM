#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import torch

# 定义全局变量
OUTPUT_DIR = './output'

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='MNIST神经网络训练和测试')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                        help='运行模式：train(仅训练), test(仅测试), both(训练后测试)')
    parser.add_argument('--model-path', type=str, default=os.path.join(OUTPUT_DIR,'mnist_model.pth'),
                        help='模型保存或加载路径')
    parser.add_argument('--test-samples', type=int, default=10,
                        help='测试时使用的样本数量')
    parser.add_argument('--custom-image', type=str, default=None,
                        help='用于测试的自定义图像路径')
    parser.add_argument('--no-gpu', action='store_true',
                        help='禁用GPU加速（即使GPU可用）')
    return parser.parse_args()

def check_gpu():
    """
    检查GPU是否可用
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
        print(f"检测到GPU: {device_name}")
        print(f"GPU数量: {device_count}")
        print(f"CUDA版本: {torch.version.cuda}")
        return True
    else:
        print("未检测到GPU，将使用CPU进行计算")
        return False

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 创建结果目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查GPU是否可用
    has_gpu = check_gpu()
    
    if args.no_gpu and has_gpu:
        print("根据参数设置，将禁用GPU加速")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    print("\n=== 系统信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    
    # 根据选择的模式执行相应操作
    if args.mode in ['train', 'both']:
        print("\n=== 开始训练模型 ===")
        print("注意: 训练过程使用了经验回放技术来解决灾难性遗忘问题")
        from train_mnist import train_network
        
        start_time = time.time()
        weights, activation_weights, biases = train_network()
        train_time = time.time() - start_time
        
        print(f"训练完成，耗时: {train_time:.2f} 秒")
    
    if args.mode in ['test', 'both']:
        print("\n=== 开始测试模型 ===")
        from test_mnist import test_model, test_custom_image, evaluate_model
        
        # 测试MNIST测试集上的性能
        accuracy = test_model(args.model_path, num_samples=args.test_samples)
        
        # 评估整个测试集上的性能
        print("\n=== 在完整测试集上评估模型 ===")
        evaluate_model(args.model_path)
        
        # 如果提供了自定义图像，也进行测试
        if args.custom_image and os.path.exists(args.custom_image):
            print(f"\n测试自定义图像: {args.custom_image}")
            test_custom_image(args.model_path, args.custom_image)
    
    print("\n=== 所有任务完成 ===")

if __name__ == "__main__":
    main() 