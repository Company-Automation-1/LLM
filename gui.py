#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import io
from PIL import Image
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QSize, QBuffer, QIODevice
from qt.layout import Ui_MainWindow
from qt.paintboard import PaintBoard
import numpy as np
import os
from my_custom_net import neural_network

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._init_ui()
        # 初始化网络参数（随机，后续可替换为训练好的参数）
        self.input_size = 784
        self.output_size = 10
        self.max_power = 4
        np.random.seed(42)
        self.weights = np.random.randn(self.output_size, self.input_size)
        self.activation_weights = np.random.randn(self.output_size, self.max_power)
        self.biases = np.random.randn(self.output_size, 1)

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
        self.pbtPredict_2.clicked.connect(self.submit_correct_result)
        
        # 初始化界面
        self.pbtClear_Callback()
        self.lbResult.setText("")
        self.lbCofidence.setText("")
        self.lbCofidence_2.setText("")

    def _center_window(self):
        """窗口居中显示"""
        frame_pos = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_pos.moveCenter(screen_center)
        self.move(frame_pos.topLeft())

    def pbtClear_Callback(self):
        """清空画板和结果显示"""
        self.paintBoard.Clear()
        self.lbResult.setText("")
        self.lbCofidence.setText("")
        self.lbCofidence_2.setText("")

    def pbtPredict_Callback(self):
        """识别按钮回调函数"""
        img_array = self.save_paintboard_image()
        if img_array is None:
            self.lbResult.setText("")
            self.lbCofidence.setText("")
            self.lbCofidence_2.setText("")
            return
        # 数据格式转换：取出图片数据，展平成(784, 1)向量
        x = img_array.reshape(784, 1)
        # 神经网络前向推理
        output = neural_network(x, self.weights, self.activation_weights, self.biases)  # (10, 1)
        print(output)
        # softmax概率分布
        probs = output.ravel()
        print(probs)
        # 预测类别
        pred = int(np.argmax(probs))
        print(pred)
        # 置信度（最大概率）
        confidence = float(np.max(probs))
        print(confidence)
        # 计算loss（用预测结果one-hot）
        t = np.zeros_like(probs)
        t[pred] = 1
        loss = -np.sum(t * np.log(probs + 1e-7))
        print(loss)
        # 更新UI显示
        self.lbResult.setText(str(pred))
        self.lbCofidence.setText(f"{confidence:.8f}")
        self.lbCofidence_2.setText(f"{loss:.6f}")

    def submit_correct_result(self):
        """提交正确结果按钮回调函数"""
        correct_result = self.lineEdit.text()
        if correct_result:
            print(f"用户提交的正确结果: {correct_result}")
            # 这里可以添加保存正确结果的逻辑
        else:
            print("请输入正确结果")

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

        """ # 保存文件
        # 创建output文件夹
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # 生成基础文件名
        filename = f'output_{int(time.time()*1000)}'
        
        # # 保存NPY数组
        # npy_filepath = os.path.join(output_dir, f'{filename}.npy')
        # np.save(npy_filepath, img_array)
        # print(f'图片数组已保存为{npy_filepath}')
        # print(f'图片归一化数组部分内容:\n', img_array[:, :, :])

        # 保存图像
        img_filepath = os.path.join(output_dir, f'{filename}.png')
        pil_img.save(img_filepath)
        print(f'图片已保存为{img_filepath}')
        """
        
        return img_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_()) 