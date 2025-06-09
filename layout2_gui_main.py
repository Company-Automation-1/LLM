#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import io
import os
from PIL import Image
from PyQt5.QtWidgets import (QMainWindow, QDesktopWidget, QApplication, QMessageBox, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                             QLineEdit, QSlider, QComboBox, QStatusBar, QFileDialog)
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtCore import QSize, QBuffer, QIODevice, Qt
from qt.paintboard import PaintBoard
import numpy as np

# 导入幂激活网络模型
from power_activation_network import PowerActivationNetwork

class ModernGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "幂激活网络手写数字识别系统"
        self.width = 900
        self.height = 700
        self.initUI()
        self.initModel()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, self.width, self.height)
        self._center_window()
        
        # 创建主窗口部件和布局
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        
        # 创建主布局
        mainLayout = QHBoxLayout()
        
        # 左侧控制面板
        leftPanel = QWidget()
        leftLayout = QVBoxLayout()
        
        # 添加标题
        titleLabel = QLabel("幂激活网络手写数字识别")
        titleLabel.setFont(QFont("Arial", 16, QFont.Bold))
        titleLabel.setAlignment(Qt.AlignCenter)
        leftLayout.addWidget(titleLabel)
        
        # 添加说明信息
        infoLabel = QLabel("使用说明：\n1. 在右侧画板上手写数字\n2. 点击'识别'按钮进行识别\n3. 如果识别错误，可输入正确结果并提交")
        infoLabel.setWordWrap(True)
        leftLayout.addWidget(infoLabel)
        
        # 添加分隔
        leftLayout.addSpacing(20)
        
        # 添加网络设置部分
        settingsLabel = QLabel("网络设置")
        settingsLabel.setFont(QFont("Arial", 12, QFont.Bold))
        leftLayout.addWidget(settingsLabel)
        
        # 隐藏层大小设置
        hiddenSizeLayout = QHBoxLayout()
        hiddenSizeLabel = QLabel("隐藏层大小:")
        self.hiddenSizeSlider = QSlider(Qt.Horizontal)
        self.hiddenSizeSlider.setMinimum(10)
        self.hiddenSizeSlider.setMaximum(200)
        self.hiddenSizeSlider.setValue(50)
        self.hiddenSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.hiddenSizeSlider.setTickInterval(10)
        self.hiddenSizeValueLabel = QLabel("50")
        self.hiddenSizeSlider.valueChanged.connect(self.updateHiddenSizeLabel)
        
        hiddenSizeLayout.addWidget(hiddenSizeLabel)
        hiddenSizeLayout.addWidget(self.hiddenSizeSlider)
        hiddenSizeLayout.addWidget(self.hiddenSizeValueLabel)
        leftLayout.addLayout(hiddenSizeLayout)
        
        # 学习率设置
        learningRateLayout = QHBoxLayout()
        learningRateLabel = QLabel("学习率:")
        self.learningRateCombo = QComboBox()
        learning_rates = ["0.001", "0.005", "0.01", "0.05", "0.1"]
        self.learningRateCombo.addItems(learning_rates)
        self.learningRateCombo.setCurrentIndex(2)  # 默认为0.01
        
        learningRateLayout.addWidget(learningRateLabel)
        learningRateLayout.addWidget(self.learningRateCombo)
        leftLayout.addLayout(learningRateLayout)
        
        # 添加分隔
        leftLayout.addSpacing(20)
        
        # 添加操作按钮
        operationLabel = QLabel("操作")
        operationLabel.setFont(QFont("Arial", 12, QFont.Bold))
        leftLayout.addWidget(operationLabel)
        
        # 识别按钮
        self.predictButton = QPushButton("识别")
        self.predictButton.setMinimumHeight(40)
        self.predictButton.clicked.connect(self.onPredictButtonClicked)
        leftLayout.addWidget(self.predictButton)
        
        # 清除按钮
        self.clearButton = QPushButton("清除")
        self.clearButton.clicked.connect(self.onClearButtonClicked)
        leftLayout.addWidget(self.clearButton)
        
        # 添加分隔
        leftLayout.addSpacing(20)
        
        # 添加反馈学习部分
        feedbackLabel = QLabel("反馈学习")
        feedbackLabel.setFont(QFont("Arial", 12, QFont.Bold))
        leftLayout.addWidget(feedbackLabel)
        
        # 正确结果输入
        correctLabelLayout = QHBoxLayout()
        correctLabelPrompt = QLabel("正确结果:")
        self.correctLabelInput = QLineEdit()
        self.correctLabelInput.setMaxLength(1)
        self.correctLabelInput.setPlaceholderText("0-9")
        
        correctLabelLayout.addWidget(correctLabelPrompt)
        correctLabelLayout.addWidget(self.correctLabelInput)
        leftLayout.addLayout(correctLabelLayout)
        
        # 提交按钮
        self.submitButton = QPushButton("提交正确结果")
        self.submitButton.clicked.connect(self.onSubmitButtonClicked)
        leftLayout.addWidget(self.submitButton)
        
        # 添加分隔
        leftLayout.addSpacing(20)
        
        # 添加模型管理部分
        modelLabel = QLabel("模型管理")
        modelLabel.setFont(QFont("Arial", 12, QFont.Bold))
        leftLayout.addWidget(modelLabel)
        
        # 保存模型按钮
        self.saveModelButton = QPushButton("保存模型")
        self.saveModelButton.clicked.connect(self.onSaveModelButtonClicked)
        leftLayout.addWidget(self.saveModelButton)
        
        # 加载模型按钮
        self.loadModelButton = QPushButton("加载模型")
        self.loadModelButton.clicked.connect(self.onLoadModelButtonClicked)
        leftLayout.addWidget(self.loadModelButton)
        
        # 重新训练按钮
        self.trainButton = QPushButton("重新训练")
        self.trainButton.clicked.connect(self.onTrainButtonClicked)
        leftLayout.addWidget(self.trainButton)
        
        # 弹性空间
        leftLayout.addStretch(1)
        
        leftPanel.setLayout(leftLayout)
        
        # 右侧画板和结果显示
        rightPanel = QWidget()
        rightLayout = QVBoxLayout()
        
        # 添加画板标签
        drawingLabel = QLabel("手写输入区域")
        drawingLabel.setAlignment(Qt.AlignCenter)
        rightLayout.addWidget(drawingLabel)
        
        # 添加画板
        self.paintBoard = PaintBoard(self, Size=QSize(280, 280), Fill=QColor(255, 255, 255, 255))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 255))
        self.paintBoard.setPenThickness(18)
        rightLayout.addWidget(self.paintBoard)
        
        # 添加分隔
        rightLayout.addSpacing(20)
        
        # 添加识别结果显示
        resultLabel = QLabel("识别结果")
        resultLabel.setAlignment(Qt.AlignCenter)
        rightLayout.addWidget(resultLabel)
        
        # 大数字显示识别结果
        self.resultDisplay = QLabel("?")
        self.resultDisplay.setFont(QFont("Arial", 72, QFont.Bold))
        self.resultDisplay.setAlignment(Qt.AlignCenter)
        self.resultDisplay.setStyleSheet("background-color: #f0f0f0; border-radius: 10px;")
        self.resultDisplay.setMinimumHeight(120)
        rightLayout.addWidget(self.resultDisplay)
        
        # 置信度显示
        confidenceLayout = QHBoxLayout()
        confidenceLabel = QLabel("置信度:")
        self.confidenceDisplay = QLabel("0.0")
        confidenceLayout.addWidget(confidenceLabel)
        confidenceLayout.addWidget(self.confidenceDisplay)
        rightLayout.addLayout(confidenceLayout)
        
        # 弹性空间
        rightLayout.addStretch(1)
        
        rightPanel.setLayout(rightLayout)
        
        # 设置左右两侧比例
        mainLayout.addWidget(leftPanel, 40)
        mainLayout.addWidget(rightPanel, 60)
        
        self.centralWidget.setLayout(mainLayout)
        
        # 添加状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
    
    def initModel(self):
        """初始化神经网络模型"""
        hidden_size = self.hiddenSizeSlider.value()
        self.network = PowerActivationNetwork(input_size=784, hidden_size=hidden_size, output_size=10)
        
        # 设置学习率
        learning_rate = float(self.learningRateCombo.currentText())
        self.network.learning_rate = learning_rate
        
        # 尝试加载已有模型
        model_path = "power_network_params.pkl"
        if os.path.exists(model_path):
            success = self.network.load_params(model_path)
            if success:
                self.statusBar.showMessage("模型加载成功")
            else:
                self.statusBar.showMessage("模型加载失败，使用初始化模型")
        else:
            self.statusBar.showMessage("未找到预训练模型，使用初始化模型")
    
    def _center_window(self):
        """将窗口居中显示"""
        frame_pos = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_pos.moveCenter(screen_center)
        self.move(frame_pos.topLeft())
    
    def updateHiddenSizeLabel(self, value):
        """更新隐藏层大小标签"""
        self.hiddenSizeValueLabel.setText(str(value))
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(self, '确认退出',
            "是否保存模型并退出程序？", QMessageBox.Yes | 
            QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            # 保存模型
            self.saveModel()
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()
    
    def onPredictButtonClicked(self):
        """识别按钮点击事件"""
        img_array = self.savePaintboardImage()
        
        if img_array is not None:
            # 使用幂激活网络进行预测
            probabilities, predicted_class = self.network.predict(img_array)
            
            # 获取置信度
            confidence = float(probabilities[predicted_class])
            
            # 更新UI显示
            self.resultDisplay.setText(str(predicted_class))
            self.confidenceDisplay.setText(f"{confidence:.8f}")
            
            self.statusBar.showMessage("识别完成")
        else:
            self.statusBar.showMessage("请先在画板上绘制数字")
    
    def onClearButtonClicked(self):
        """清除按钮点击事件"""
        self.paintBoard.Clear()
        self.resultDisplay.setText("?")
        self.confidenceDisplay.setText("0.0")
        self.statusBar.showMessage("已清除")
    
    def onSubmitButtonClicked(self):
        """提交正确结果按钮点击事件"""
        correct_result = self.correctLabelInput.text()
        img_array = self.savePaintboardImage(save_to_disk=False)
        
        if correct_result and img_array is not None:
            try:
                correct_label = int(correct_result)
                if 0 <= correct_label <= 9:
                    # 更新网络权重
                    self.network.update_weights(img_array, correct_label)
                    self.statusBar.showMessage(f"已更新模型: 正确结果为 {correct_label}")
                    
                    # 保存训练样本
                    output_dir = "training_samples"
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f'sample_{correct_label}_{int(time.time()*1000)}'
                    npy_filepath = os.path.join(output_dir, f'{filename}.npy')
                    np.save(npy_filepath, img_array)
                    
                    # 保存为图像
                    img_filepath = os.path.join(output_dir, f'{filename}.png')
                    pil_img = Image.fromarray((img_array[0, 0] * 255).astype(np.uint8))
                    pil_img.save(img_filepath)
                    
                    # 清空输入框
                    self.correctLabelInput.setText("")
                else:
                    self.statusBar.showMessage("请输入0-9之间的数字")
            except ValueError:
                self.statusBar.showMessage("请输入有效的数字")
        else:
            self.statusBar.showMessage("请先绘制数字并输入正确结果")
    
    def onSaveModelButtonClicked(self):
        """保存模型按钮点击事件"""
        self.saveModel(ask_path=True)
    
    def onLoadModelButtonClicked(self):
        """加载模型按钮点击事件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "", "Pickle Files (*.pkl);;All Files (*)", options=options
        )
        
        if file_path:
            success = self.network.load_params(file_path)
            if success:
                self.statusBar.showMessage(f"模型从 {file_path} 加载成功")
                QMessageBox.information(self, "加载成功", f"模型已从 {file_path} 成功加载")
            else:
                self.statusBar.showMessage(f"模型从 {file_path} 加载失败")
                QMessageBox.warning(self, "加载失败", f"无法从 {file_path} 加载模型")
    
    def onTrainButtonClicked(self):
        """重新训练按钮点击事件"""
        reply = QMessageBox.question(self, '确认训练',
            "训练可能需要一段时间，确定要开始训练吗？", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.statusBar.showMessage("正在准备训练...")
            
            # 获取当前设置
            hidden_size = self.hiddenSizeSlider.value()
            learning_rate = float(self.learningRateCombo.currentText())
            
            # 创建命令
            command = f"python train_power_network.py --epochs 5 --learning-rate {learning_rate} --hidden-size {hidden_size}"
            
            # 在新进程中启动训练
            import subprocess
            try:
                self.statusBar.showMessage(f"开始训练: {command}")
                subprocess.Popen(command, shell=True)
                QMessageBox.information(self, "训练开始", "训练已在后台开始，请等待完成。\n训练完成后，可使用'加载模型'按钮加载新模型。")
            except Exception as e:
                self.statusBar.showMessage(f"训练启动失败: {str(e)}")
                QMessageBox.warning(self, "训练失败", f"无法启动训练: {str(e)}")
    
    def saveModel(self, ask_path=False):
        """保存当前模型"""
        if ask_path:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存模型", "", "Pickle Files (*.pkl);;All Files (*)", options=options
            )
            
            if not file_path:
                return
        else:
            file_path = "power_network_params.pkl"
        
        try:
            self.network.save_params(file_path)
            self.statusBar.showMessage(f"模型已保存到 {file_path}")
            if ask_path:
                QMessageBox.information(self, "保存成功", f"神经网络模型已成功保存到 {file_path}")
        except Exception as e:
            self.statusBar.showMessage(f"模型保存失败: {str(e)}")
            if ask_path:
                QMessageBox.warning(self, "保存失败", f"模型保存失败: {str(e)}")
    
    def savePaintboardImage(self, save_to_disk=True):
        """将画板内容保存为28x28 PNG图片，并输出归一化数组"""
        __img = self.paintBoard.getContentAsQImage()
        # 判断画板是否为空（全白）
        if __img is None or __img.isNull():
            self.statusBar.showMessage("画板为空，请先绘制数字")
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
            
            # 保存图像
            img_filepath = os.path.join(output_dir, f'{filename}.png')
            pil_img.save(img_filepath)
            
            self.statusBar.showMessage(f"图像已保存为 {img_filepath}")
        
        return img_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式，看起来更现代
    gui = ModernGUI()
    gui.show()
    sys.exit(app.exec_())