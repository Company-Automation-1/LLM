#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QSize

class PaintBoard(QWidget):
    def __init__(self, Parent=None, Size=QSize(320, 240), Fill=QColor(255,255,255,255)):
        super().__init__(Parent)
    
        # 初始化参数
        self.__size = Size                  # 画板尺寸
        self.__fill = Fill                  # 画板默认填充颜色

        self.__thickness = 18               # 默认画笔粗细
        self.__penColor = QColor(0,0,0,255)   # 默认画笔颜色

        self.__begin_point = QPoint()
        self.__end_point = QPoint()

        # 初始化画板界面
        self.__board = QPixmap(self.__size)
        self.__board.fill(Fill) 
        self.setFixedSize(self.__size)
        # self.__painter = QPainter()  # 移除类成员QPainter，改为临时创建

    # 清空画板
    def Clear(self):
        self.__board.fill(self.__fill)
        self.update()

    def setBoardFill(self, fill):
        self.__fill = fill
        self.__board.fill(fill)
        self.update()
    
    # 设置画笔颜色    
    def setPenColor(self, color):
        self.__penColor = color

    # 设置画笔粗细    
    def setPenThickness(self, thickness=10):    
        self.__thickness = thickness  

    # 获取画板QImage类型图片
    def getContentAsQImage(self):
        image = self.__board.toImage()
        return image 

    # 双缓冲绘图，绘图事件
    def paintEvent(self, paintEvent):         
        painter = QPainter(self)  # 临时创建QPainter
        painter.drawPixmap(0, 0, self.__board)
        # painter.end() 不需要手动end，QPainter对象销毁时自动处理

    def mousePressEvent(self, mouseEvent):
        if mouseEvent.button() == Qt.LeftButton:
            self.__begin_point = mouseEvent.pos()
            self.__end_point = self.__begin_point
            # self.update()

    def mouseMoveEvent(self, mouseEvent):
        if mouseEvent.buttons() == Qt.LeftButton:
            self.__end_point = mouseEvent.pos()

            # 画入缓冲区
            painter = QPainter(self.__board)  # 临时创建QPainter
            painter.setPen(QPen(self.__penColor, self.__thickness))  
            painter.drawLine(self.__begin_point, self.__end_point)
            # painter.end() 不需要手动end

            self.__begin_point = self.__end_point
            self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = PaintBoard()
    demo.show()
    sys.exit(app.exec_())