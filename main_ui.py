# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1027, 614)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.select_pb = QtWidgets.QPushButton(self.centralwidget)
        self.select_pb.setGeometry(QtCore.QRect(420, 210, 161, 71))
        self.select_pb.setObjectName("select_pb")
        self.convert_pb = QtWidgets.QPushButton(self.centralwidget)
        self.convert_pb.setGeometry(QtCore.QRect(420, 330, 161, 71))
        self.convert_pb.setObjectName("convert_pb")
        self.img_before = QtWidgets.QLabel(self.centralwidget)
        self.img_before.setGeometry(QtCore.QRect(30, 100, 341, 421))
        self.img_before.setStyleSheet("border: 1px solid black;")
        self.img_before.setText("")
        self.img_before.setScaledContents(True)
        self.img_before.setObjectName("img_before")
        self.img_after = QtWidgets.QLabel(self.centralwidget)
        self.img_after.setGeometry(QtCore.QRect(630, 100, 341, 421))
        self.img_after.setStyleSheet("border: 1px solid black;")
        self.img_after.setText("")
        self.img_after.setScaledContents(True)
        self.img_after.setObjectName("img_after")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 30, 131, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLineWidth(9)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(750, 30, 131, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setLineWidth(9)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1027, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.select_pb.setText(_translate("MainWindow", "Select"))
        self.convert_pb.setText(_translate("MainWindow", "Convert"))
        self.label.setText(_translate("MainWindow", "ORIGINAL"))
        self.label_2.setText(_translate("MainWindow", "CONVERTED"))
