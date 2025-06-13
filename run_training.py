#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time

def clear_screen():
    """清除控制台屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印程序标题"""
    print("="*80)
    print("                      MNIST训练性能分析工具")
    print("="*80)
    print("该工具可以帮助您分析神经网络训练过程中的性能瓶颈。")
    print("通过不同的训练配置，您可以观察各个函数的执行时间和调用次数。")
    print("="*80)
    print()

def print_menu():
    """打印菜单选项"""
    print("请选择训练配置：")
    print("1. 超快测试 (10样本, 1轮, 小网络)")
    print("2. 标准测试 (5000样本, 5轮)")
    print("3. 完整测试 (所有样本, 10轮)")
    print("4. 自定义配置")
    print("5. 查看上次训练的性能报告")
    print("0. 退出")
    print()
    return input("请输入选项 [0-5]: ")

def run_training(config):
    """执行训练脚本"""
    cmd = [sys.executable, 'src/train_mnist_with_tracking.py']
    
    # 添加命令行参数
    for key, value in config.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    # 执行命令
    print(f"执行命令: {' '.join(cmd)}")
    print("="*80)
    print("开始训练...\n")
    
    try:
        # 创建进程
        process = subprocess.Popen(cmd)
        
        # 等待进程完成
        process.wait()
        print("\n训练完成！")
            
    except KeyboardInterrupt:
        print("\n训练已中断！")
        # 尝试终止进程
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"\n发生错误: {e}")
    
    input("\n按回车键继续...")

def view_report():
    """查看上次训练的性能报告"""
    clear_screen()
    print("="*80)
    print("                      Performance Report Viewer")
    print("="*80)
    
    # 检查报告文件是否存在
    report_file = '/results/performance_report.csv'
    if not os.path.exists(report_file):
        print("Performance report file not found! Please run training first.")
        input("\nPress Enter to continue...")
        return
    
    # 读取并显示报告内容
    try:
        import pandas as pd
        df = pd.read_csv(report_file)
        print("Performance Summary (Top 10 functions):")
        print(df.head(10).to_string(index=False))
        
        # 显示图表信息
        chart_file = '/results/performance_charts.png'
        if os.path.exists(chart_file):
            print(f"\nPerformance chart saved at: {os.path.abspath(chart_file)}")
            
            # 尝试打开图表文件
            try:
                if os.name == 'nt':  # Windows
                    os.system(f'start {chart_file}')
                elif os.name == 'posix':  # macOS 或 Linux
                    if sys.platform == 'darwin':  # macOS
                        os.system(f'open {chart_file}')
                    else:  # Linux
                        os.system(f'xdg-open {chart_file}')
            except:
                print("Cannot open chart file automatically. Please open it manually.")
        
    except Exception as e:
        print(f"Error reading report: {e}")
    
    input("\nPress Enter to continue...")

def custom_config():
    """让用户输入自定义配置"""
    clear_screen()
    print("="*80)
    print("                      自定义训练配置")
    print("="*80)
    
    config = {}
    
    try:
        # 获取样本数量
        samples = input("训练样本数量 (-1表示全部，默认为5000): ")
        config['samples'] = int(samples) if samples.strip() else 5000
        
        # 获取批次大小
        batch_size = input("批次大小 (默认为64): ")
        config['batch-size'] = int(batch_size) if batch_size.strip() else 64
        
        # 获取训练轮数
        epochs = input("训练轮数 (默认为5): ")
        config['epochs'] = int(epochs) if epochs.strip() else 5
        
        # 获取学习率
        learning_rate = input("初始学习率 (默认为0.01): ")
        config['learning-rate'] = float(learning_rate) if learning_rate.strip() else 0.01
        
        # 获取隐藏层大小
        hidden_size = input("隐藏层大小 (默认为100): ")
        config['hidden-size'] = int(hidden_size) if hidden_size.strip() else 100
        
        return config
    
    except ValueError:
        print("输入格式错误！将使用默认配置。")
        input("\n按回车键继续...")
        return {
            'samples': 5000,
            'batch-size': 64,
            'epochs': 5,
            'learning-rate': 0.01,
            'hidden-size': 100
        }

def main():
    """主函数"""
    while True:
        clear_screen()
        print_header()
        choice = print_menu()
        
        if choice == '0':
            print("感谢使用！再见！")
            break
            
        elif choice == '1':
            # 快速测试
            config = {
                'samples': 10,
                'epochs': 1,
                'batch-size': 10,  # 减小批次大小
                'learning-rate': 0.01,
                'hidden-size': 10  # 使用更小的隐藏层
            }
            run_training(config)
            
        elif choice == '2':
            # 标准测试
            config = {
                'samples': 5000,
                'epochs': 5,
                'batch-size': 64,
                'learning-rate': 0.01,
                'hidden-size': 100
            }
            run_training(config)
            
        elif choice == '3':
            # 完整测试
            config = {
                'samples': -1,  # 全部样本
                'epochs': 10,
                'batch-size': 64,
                'learning-rate': 0.01,
                'hidden-size': 100
            }
            run_training(config)
            
        elif choice == '4':
            # 自定义配置
            config = custom_config()
            if config:
                run_training(config)
                
        elif choice == '5':
            # 查看性能报告
            view_report()
            
        else:
            print("无效的选项！请重新输入。")
            time.sleep(1)

if __name__ == "__main__":
    main() 