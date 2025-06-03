# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'layout.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        # 清除数据按钮
        self.pbtClear = QtWidgets.QPushButton(MainWindow)
        self.pbtClear.setGeometry(QtCore.QRect(80, 440, 120, 30))
        self.pbtClear.setObjectName("pbtClear")

        # 输出图片按钮
        self.pbtPredict = QtWidgets.QPushButton(MainWindow)
        self.pbtPredict.setGeometry(QtCore.QRect(80, 500, 120, 30))
        self.pbtPredict.setObjectName("pbtPredict")

        # # 数据显示区域
        # self.lbDataArea = QtWidgets.QLabel(MainWindow)
        # self.lbDataArea.setGeometry(QtCore.QRect(540, 350, 224, 224))
        # self.lbDataArea.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.lbDataArea.setFrameShape(QtWidgets.QFrame.Box)
        # self.lbDataArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.lbDataArea.setLineWidth(4)
        # self.lbDataArea.setMidLineWidth(0)
        # self.lbDataArea.setText("")
        # self.lbDataArea.setObjectName("lbDataArea")

        # # 垂直布局容器
        # self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        # self.verticalLayoutWidget.setGeometry(QtCore.QRect(540, 350, 221, 221))
        # self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        # self.dArea_Layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        # self.dArea_Layout.setContentsMargins(0, 0, 0, 0)
        # self.dArea_Layout.setSpacing(0)
        # self.dArea_Layout.setObjectName("dArea_Layout")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图片处理视口"))
        self.pbtClear.setText(_translate("MainWindow", "清除数据"))
        self.pbtPredict.setText(_translate("MainWindow", "输出图片"))
