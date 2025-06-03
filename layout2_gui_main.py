#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import io
from PIL import Image
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QSize, QBuffer, QIODevice
from qt.layout2 import Ui_MainWindow
from qt.paintboard import PaintBoard
import numpy as np
import os

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._init_ui()

    def _init_ui(self):
        self._center_window()
        # # 隐藏原有lbDataArea，只用自定义画板
        # self.lbDataArea.hide()
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(255, 255, 255, 255))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 255))
        self.paintBoard.setGeometry(540, 350, 224, 224)
        self.paintBoard.setParent(self)
        self.paintBoard.show()
        self.pbtPredict.clicked.connect(self.save_paintboard_image)
        self.pbtClear.clicked.connect(self.clear_data_area)
        self.clear_data_area()

    def _center_window(self):
        """窗口居中显示"""
        frame_pos = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_pos.moveCenter(screen_center)
        self.move(frame_pos.topLeft())

    def clear_data_area(self):
        """清空画板内容"""
        self.paintBoard.Clear()

    def save_paintboard_image(self):
        """将画板内容保存为28x28 PNG图片，并输出归一化数组"""
        qimg = self.paintBoard.getContentAsQImage()
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qimg.save(buffer, "PNG")
        pil_img = Image.open(io.BytesIO(buffer.data()))

        pil_img = pil_img.resize((28, 28), Image.LANCZOS)  # 缩放
        pil_img = pil_img.convert('L')  # 转为灰度

        # 转为 numpy 数组并归一化
        img_array = np.array(pil_img, dtype=np.float32).reshape(1, 1, 28, 28) / 255.0

        # 创建output文件夹
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # 生成基础文件名
        filename = f'output_{int(time.time()*1000)}'
        
        # # 保存PNG图片
        # png_filepath = os.path.join(output_dir, f'{filename}.png')
        # pil_img.save(png_filepath)
        # print(f'图片已保存为{png_filepath}')
        
        # 保存NPY数组
        npy_filepath = os.path.join(output_dir, f'{filename}.npy')
        np.save(npy_filepath, img_array)
        print(f'图片数组已保存为{npy_filepath}')
        print(f'图片归一化数组部分内容:\n', img_array[:, :, :])
        
        # # 保存TXT文本
        # txt_filepath = os.path.join(output_dir, f'{filename}.txt')
        # np.savetxt(txt_filepath, img_array[0, 0], fmt='%.4f')
        # print(f'图片归一化数组文本已保存为{txt_filepath}')
        
        return img_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())