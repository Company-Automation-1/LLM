#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import io
import os
from PIL import Image
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication, QMessageBox, QPushButton
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QSize, QBuffer, QIODevice
from qt.layout import Ui_MainWindow
from qt.paintboard import PaintBoard
import numpy as np

# 导入幂激活网络模型
from power_activation_network import PowerActivationNetwork

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._init_ui()
        self._init_model()

    def _init_ui(self):
        """初始化界面"""
        self._center_window()
        # 创建画板并设置属性
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(255, 255, 255, 255))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 255))
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
        self.lbCofidence.setText("")
        
        # 添加保存模型按钮
        try:
            self.pbtSaveModel = self.findChild(QPushButton, "pbtSaveModel")
            if self.pbtSaveModel:
                self.pbtSaveModel.clicked.connect(self.save_model)
        except:
            print("未找到保存模型按钮")

    def _init_model(self):
        """初始化神经网络模型"""
        self.network = PowerActivationNetwork(input_size=784, hidden_size=50, output_size=10)
        
        # 尝试加载已有模型
        model_path = "models\power_network_epoch1.pkl"
        if os.path.exists(model_path):
            success = self.network.load_params(model_path)
            if success:
                self.statusBar().showMessage("模型加载成功")
            else:
                self.statusBar().showMessage("模型加载失败，使用初始化模型")
        else:
            self.statusBar().showMessage("未找到预训练模型，使用初始化模型")

    def _center_window(self):
        """窗口居中显示"""
        frame_pos = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_pos.moveCenter(screen_center)
        self.move(frame_pos.topLeft())
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(self, '确认退出',
            "确定要退出程序吗？", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 保存模型
            self.save_model(show_message=False)
            event.accept()
        else:
            event.ignore()

    def pbtClear_Callback(self):
        """清空画板和结果显示"""
        self.paintBoard.Clear()
        self.lbResult.setText("")
        self.lbCofidence.setText("")

    def pbtPredict_Callback(self):
        """识别按钮回调函数"""
        img_array = self.save_paintboard_image()
        
        if img_array is not None:
            # 使用幂激活网络进行预测
            probabilities, predicted_class = self.network.predict(img_array)
            
            # 获取置信度
            confidence = float(probabilities[predicted_class])
            
            # 更新UI显示
            self.lbResult.setText(str(predicted_class))
            self.lbCofidence.setText(f"{confidence:.8f}")
            
            self.statusBar().showMessage("识别完成")
        else:
            self.statusBar().showMessage("请先在画板上绘制数字")

    def submit_correct_result(self):
        """提交正确结果按钮回调函数"""
        correct_result = self.lineEdit.text()
        img_array = self.save_paintboard_image(save_to_disk=False)
        
        if correct_result and img_array is not None:
            try:
                correct_label = int(correct_result)
                if 0 <= correct_label <= 9:
                    # 更新网络权重
                    self.network.update_weights(img_array, correct_label)
                    self.statusBar().showMessage(f"已更新模型: 正确结果为 {correct_label}")
                    
                    # 保存训练样本
                    output_dir = "training_samples"
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f'sample_{correct_label}_{int(time.time()*1000)}'
                    npy_filepath = os.path.join(output_dir, f'{filename}.npy')
                    np.save(npy_filepath, img_array)
                    
                    # 清空输入框
                    self.lineEdit.setText("")
                else:
                    self.statusBar().showMessage("请输入0-9之间的数字")
            except ValueError:
                self.statusBar().showMessage("请输入有效的数字")
        else:
            self.statusBar().showMessage("请先绘制数字并输入正确结果")

    def save_model(self, show_message=True):
        """保存当前模型"""
        try:
            self.network.save_params()
            self.statusBar().showMessage("模型已保存")
            if show_message:
                QMessageBox.information(self, "保存成功", "神经网络模型已成功保存")
        except Exception as e:
            self.statusBar().showMessage(f"模型保存失败: {str(e)}")
            if show_message:
                QMessageBox.warning(self, "保存失败", f"模型保存失败: {str(e)}")

    def save_paintboard_image(self, save_to_disk=True):
        """将画板内容保存为28x28 PNG图片，并输出归一化数组"""
        __img = self.paintBoard.getContentAsQImage()
        # 判断画板是否为空（全白）
        if __img is None or __img.isNull():
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

        if save_to_disk:
            # 创建output文件夹
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            # 生成基础文件名
            filename = f'output_{int(time.time()*1000)}'
            
            # 保存NPY数组
            npy_filepath = os.path.join(output_dir, f'{filename}.npy')
            np.save(npy_filepath, img_array)
            print(f'图片数组已保存为{npy_filepath}')
            print(f'图片归一化数组部分内容:\n', img_array[:, :, :])
            
            # 保存图像
            img_filepath = os.path.join(output_dir, f'{filename}.png')
            pil_img.save(img_filepath)
            print(f'图片已保存为{img_filepath}')
        
        return img_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_()) 