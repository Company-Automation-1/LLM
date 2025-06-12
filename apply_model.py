#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from PyQt5.QtWidgets import QApplication

from main import MainWindow
from model import load_model
from util import pt

def apply_model_to_gui(model_path):
    """
    将训练好的模型应用到GUI界面
    
    参数:
    - model_path: 模型文件路径
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        pt("错误", f"模型文件 {model_path} 不存在", color='red')
        return
    
    # 加载模型
    try:
        model_params = load_model(model_path)
        pt("模型加载成功", model_path)
    except Exception as e:
        pt("模型加载失败", str(e), color='red')
        return
    
    # 创建应用
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # 更新GUI中的网络参数
    window.old_weights = model_params['weights']
    window.old_activation_weights = model_params['activation_weights']
    window.old_biases = model_params['biases']
    
    # 连接预测按钮
    window.pbtPredict.clicked.connect(window.pbtPredict_Callback)
    
    # 显示GUI
    window.show()
    pt("GUI已启动", "请在界面上绘制数字进行识别")
    
    # 运行应用
    sys.exit(app.exec_())

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = './models/final_model.pkl'
        pt("使用默认模型路径", model_path)
    
    # 应用模型到GUI
    apply_model_to_gui(model_path)

if __name__ == "__main__":
    main() 