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

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._initUI()

    def _initUI(self):
        self.center()
        # 隐藏原有lbDataArea，只用自定义画板
        self.lbDataArea.hide()
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(255, 255, 255, 255))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 255))
        self.paintBoard.setGeometry(540, 350, 224, 224)
        self.paintBoard.setParent(self)
        self.paintBoard.show()
        self.pbtPredict.clicked.connect(self.save_paintboard_image)
        self.pbtClear.clicked.connect(self.clearDataArea)
        self.clearDataArea()

    def center(self):
        """窗口居中显示"""
        framePos = self.frameGeometry()
        scPos = QDesktopWidget().availableGeometry().center()
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    def clearDataArea(self):
        """清空画板内容"""
        self.paintBoard.Clear()

    def save_paintboard_image(self):
        """将画板内容保存为28x28 PNG图片，并输出归一化数组"""
        qimg = self.paintBoard.getContentAsQImage()
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qimg.save(buffer, "PNG")
        pil_img = Image.open(io.BytesIO(buffer.data()))
        resample = Image.LANCZOS
        pil_img = pil_img.resize((28, 28), resample)
        # 转为灰度
        pil_img = pil_img.convert('L')
        # 转为 numpy 数组并归一化
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 28, 28)
        # 创建output文件夹
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        # 保存图片和npy到output文件夹
        filename = f'output_{int(time.time()*1000)}.png'
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath)
        print(f'图片已保存为{filepath}')
        npy_filename = filename.replace('.png', '.npy')
        npy_filepath = os.path.join(output_dir, npy_filename)
        np.save(npy_filepath, img_array)
        print(f'图片数组已保存为{npy_filepath}')
        print(f'图片归一化数组部分内容:\n', img_array[0, :4, :4])
        return img_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()
    sys.exit(app.exec_())