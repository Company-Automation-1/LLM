#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import io
import numpy as np
from PIL import Image
import torch
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QSize, QBuffer, QIODevice

# 布局模块
from qt.layout import Ui_MainWindow
# 画板模块
from qt.paintboard import PaintBoard

# 导入所需模块
from util import pt
from model import save_model

# 模式选择: 'torch' 或 'numpy'
MODE = 'torch'  # 'numpy'

# 根据模式导入不同的模块
if MODE == 'numpy':
    # 导入numpy版本的训练器模块
    from trainers import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient
    from networks import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
    from activations import softplus, sigmoid
else:
    # 导入torch版本的训练器模块
    from tc_trainers import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient
    from tc_networks import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network
    from tc_activations import softplus, sigmoid

class MainWindow(QMainWindow, Ui_MainWindow):

    #TODO: 初始化函数
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._init_ui()
        # 初始化网络参数（随机，后续可替换为训练好的参数）
        self.input_size = 784
        self.output_size = 10
        self.max_power = 2
        np.random.seed(42)
        
        # 根据模式初始化参数
        if MODE == 'numpy':
            #!################ numpy
            self.weight = np.random.randn(self.output_size, self.input_size)
            self.activation_weight = np.random.randn(self.output_size, self.max_power)
            self.biases = np.random.randn(self.output_size, 1)
            
            self.old_output = []  # 预测的正确结果 one-hot
            self.old_vectors = []  # 前向传播过程中的中间向量
            self.old_intermediate_vectors = []  # 前向传播过程中的中间向量
            self.old_weights = [self.weight, np.random.randn(10, 10)]
            # self.old_activation_weights = [self.activation_weight, self.activation_weight]
            self.old_activation_weights = [np.zeros((10, 2)), np.zeros((10, 2))]
            self.old_biases = [self.biases, self.biases]
            #!################ numpy
        else:
            #!################ torch
            # 直接创建torch张量
            self.weight = torch.randn(self.output_size, self.input_size)
            self.activation_weight = torch.randn(self.output_size, self.max_power)
            self.biases = torch.randn(self.output_size, 1)
            
            self.old_output = []  # 预测的正确结果 one-hot
            self.old_vectors = []  # 前向传播过程中的中间向量
            self.old_intermediate_vectors = []  # 前向传播过程中的中间向量
            
            self.old_weights = [self.weight, torch.randn(10, 10)]
            self.old_activation_weights = [torch.zeros((10, 2)), torch.zeros((10, 2))]
            self.old_biases = [self.biases, self.biases]
            #!################ torch

        pt("初始的参数", self.old_activation_weights)
        # pt("初始参数", self.old_weights[1].shape)
        # pt("初始激活参数", self.old_activation_weights[1].shape)
        # pt("初始偏置", self.old_biases[1].shape)

    #TODO: 初始化UI
    def _init_ui(self):
        self._center_window()
        # 创建画板并设置属性
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(0, 0, 0, 255))
        self.paintBoard.setPenColor(QColor(255, 255, 255, 255))
        self.paintBoard.setGeometry(510, 210, 224, 224)
        self.paintBoard.setParent(self)
        self.paintBoard.show()
        
        # 连接按钮信号
        self.pbtClear.clicked.connect(self.pbtClear_Callback)
        self.pbtPredict.clicked.connect(self.pbtPredict_Callback)
        self.pbtPredict_2.clicked.connect(self.submit_correct_result)
        
        # 初始化界面
        self.pbtClear_Callback()
        self.lbResult.setText("")
        self.lbCofidence_2.setText("")

    #TODO: 窗口居中显示
    def _center_window(self):
        """窗口居中显示"""
        frame_pos = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_pos.moveCenter(screen_center)
        self.move(frame_pos.topLeft())

    #TODO: 清空画板和结果显示
    def pbtClear_Callback(self):
        """清空画板和结果显示"""
        self.paintBoard.Clear()
        self.lbResult.setText("")
        self.lbCofidence_2.setText("")

    #TODO: 识别按钮回调函数
    def pbtPredict_Callback(self):
        """识别按钮回调函数"""

        img_array = self.save_paintboard_image()
        if img_array is None:
            self.lbCofidence_2.setText("")
            return
        
        # 数据格式转换：取出图片数据，展平成(784, 1)向量
        if MODE == 'numpy':
            # numpy模式直接使用numpy数组
            x = img_array.reshape(784, 1)
        else:
            # torch模式转换为torch张量，保持列向量形状
            x = torch.tensor(img_array.reshape(784, 1), dtype=torch.float32)

        # 神经网络前向推理
        output, vectors, intermediate_vectors = neural_network(x, self.old_weights, self.old_activation_weights, self.old_biases, softplus, 1)
        
        # 预测类别和置信度
        if MODE == 'numpy':
            # numpy模式直接使用
            pred = int(np.argmax(output))
            confidence = float(np.max(output))
        else:
            # torch模式获取值
            pred = int(torch.argmax(output))
            confidence = float(torch.max(output).item())

        # 保存参数
        self.old_output = output
        self.old_vectors = vectors
        self.old_intermediate_vectors = intermediate_vectors
        
        # 更新UI显示
        self.lbResult.setText(str(pred))

    #TODO: 提交正确结果按钮回调函数
    def submit_correct_result(self):
        """提交正确结果按钮回调函数"""
        correct_result = self.lineEdit.text()
        learning_rate = float(self.lineEdit_2.text())

        if correct_result:
            if MODE == 'numpy':
                # numpy模式使用numpy数组
                res = np.zeros((10,1))
                res[int(correct_result)] = 1
            else:
                # torch模式创建列向量
                res = torch.zeros(10, 1, dtype=torch.float32)
                res[int(correct_result)] = 1

            # 反向传播计算梯度
            back_weights, back_act_weights, back_biases = gradient(vectors = self.old_vectors,
                     intermediate_vectors = self.old_intermediate_vectors,
                     weights = self.old_weights,
                     activation_weights = self.old_activation_weights,
                     biases = self.old_biases,
                     activation_function = softplus,
                     diff_activation = sigmoid,
                     temperature = 1,
                     actual = res
                    )
            
            # 更新参数
            layers = len(self.old_weights)
            for layer in range(layers):
                self.old_weights[layer] -= learning_rate * back_weights[layer]
                self.old_activation_weights[layer] -= learning_rate * back_act_weights[layer]
                self.old_biases[layer] -= learning_rate * back_biases[layer]

            pt("更新后的参数", self.old_activation_weights[1][:, 0])

            # 计算loss
            if MODE == 'numpy':
                # numpy模式直接计算
                loss = np.sum((self.old_output - res) * (self.old_output - res))
            else:
                # torch模式直接计算张量的MSE
                loss = torch.sum((self.old_output - res) * (self.old_output - res)).item()
                
            pt("loss", loss)
            
            # 更新UI显示
            self.lbCofidence_2.setText(f"{loss:.6f}")

        else:
            print("请输入正确结果")

    #TODO: 保存画板内容为28x28 PNG图片并输出归一化数组
    def save_paintboard_image(self):
        """将画板内容保存为28x28 PNG图片，并输出归一化数组"""
        __img = self.paintBoard.getContentAsQImage()
        # 判断画板是否为空（全白）
        if __img is None or __img.isNull() or __img == __img.copy().fill(255):
            print("画板为空，不保存图片")
            return None
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        __img.save(buffer, "PNG")
        pil_img = Image.open(io.BytesIO(buffer.data()))

        pil_img = pil_img.resize((28, 28), Image.LANCZOS)  # 缩放
        pil_img = pil_img.convert('L')  # 转为灰度

        # 转为 numpy 数组并归一化
        img_array = np.array(pil_img, dtype=np.float32).reshape(1, 1, 28, 28) / 255.0
        
        return img_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_()) 