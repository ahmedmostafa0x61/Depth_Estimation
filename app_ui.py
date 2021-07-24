# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app_ui.ui'
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet("QMainWindow{\n"
"\n"
"background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0.2 #136a8a, stop:1 #136a8a);\n"
"\n"
"}\n"
"\n"
"#2C3E50\n"
"\n"
"#4CA1AF\n"
"\n"
"\n"
"/* QLabel */\n"
"QLabel{}\n"
"/*QComboBox*/\n"
"QComboBox{}\n"
"\n"
"\n"
"/* Push Buttons */\n"
"QPushButton{border: 3px solid #8f8f91;\n"
"                    border-radius: 10px;\n"
"                    background-color: #F2F2F2;                    \n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #F2F2F2;\n"
"    border:4px solid #64b3f4;/*#00909e*/\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #2980B9, stop: 1 #4286f4);\n"
"}\n"
"\n"
"\n"
"")
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(10, 10, 10, 5)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 7, 0)
        self.verticalLayout.setSpacing(7)
        self.verticalLayout.setObjectName("verticalLayout")
        self.original_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.original_label.setFont(font)
        self.original_label.setStyleSheet("")
        self.original_label.setLineWidth(9)
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setObjectName("original_label")
        self.verticalLayout.addWidget(self.original_label)
        self.img_before = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_before.sizePolicy().hasHeightForWidth())
        self.img_before.setSizePolicy(sizePolicy)
        self.img_before.setMaximumSize(QtCore.QSize(1000, 800))
        self.img_before.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.img_before.setFont(font)
        self.img_before.setStyleSheet("border: 1px solid black;")
        self.img_before.setScaledContents(True)
        self.img_before.setAlignment(QtCore.Qt.AlignCenter)
        self.img_before.setObjectName("img_before")
        self.verticalLayout.addWidget(self.img_before)
        self.select_pb = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_pb.sizePolicy().hasHeightForWidth())
        self.select_pb.setSizePolicy(sizePolicy)
        self.select_pb.setMinimumSize(QtCore.QSize(175, 45))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_pb.setFont(font)
        self.select_pb.setObjectName("select_pb")
        self.verticalLayout.addWidget(self.select_pb, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.setStretch(1, 10)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setContentsMargins(2, 150, 2, 150)
        self.verticalLayout_3.setSpacing(9)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.convert_pb = QtWidgets.QPushButton(self.centralwidget)
        self.convert_pb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.convert_pb.sizePolicy().hasHeightForWidth())
        self.convert_pb.setSizePolicy(sizePolicy)
        self.convert_pb.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.convert_pb.setFont(font)
        self.convert_pb.setCheckable(False)
        self.convert_pb.setObjectName("convert_pb")
        self.verticalLayout_3.addWidget(self.convert_pb, 0, QtCore.Qt.AlignBottom)
        self.line = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setMinimumSize(QtCore.QSize(0, 0))
        self.line.setStyleSheet("")
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setLineWidth(5)
        self.line.setMidLineWidth(0)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.verticalLayout_3.addWidget(self.line)
        self.select_acc = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_acc.sizePolicy().hasHeightForWidth())
        self.select_acc.setSizePolicy(sizePolicy)
        self.select_acc.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.select_acc.setFont(font)
        self.select_acc.setEditable(False)
        self.select_acc.setFrame(True)
        self.select_acc.setObjectName("select_acc")
        self.select_acc.addItem("")
        self.select_acc.addItem("")
        self.verticalLayout_3.addWidget(self.select_acc, 0, QtCore.Qt.AlignTop)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(7, -1, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.convert_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.convert_label.setFont(font)
        self.convert_label.setLineWidth(9)
        self.convert_label.setAlignment(QtCore.Qt.AlignCenter)
        self.convert_label.setObjectName("convert_label")
        self.verticalLayout_2.addWidget(self.convert_label)
        self.img_after = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_after.sizePolicy().hasHeightForWidth())
        self.img_after.setSizePolicy(sizePolicy)
        self.img_after.setMaximumSize(QtCore.QSize(1000, 800))
        self.img_after.setStyleSheet("border: 1px solid black;")
        self.img_after.setText("")
        self.img_after.setScaledContents(True)
        self.img_after.setObjectName("img_after")
        self.verticalLayout_2.addWidget(self.img_after)
        self.save_pb = QtWidgets.QPushButton(self.centralwidget)
        self.save_pb.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_pb.sizePolicy().hasHeightForWidth())
        self.save_pb.setSizePolicy(sizePolicy)
        self.save_pb.setMinimumSize(QtCore.QSize(175, 45))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.save_pb.setFont(font)
        self.save_pb.setCheckable(False)
        self.save_pb.setObjectName("save_pb")
        self.verticalLayout_2.addWidget(self.save_pb, 0, QtCore.Qt.AlignHCenter)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1027, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.select_acc.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.original_label.setText(_translate("MainWindow", "ORIGINAL"))
        self.img_before.setText(_translate("MainWindow", "No Image Selected"))
        self.select_pb.setText(_translate("MainWindow", "Select Image"))
        self.convert_pb.setText(_translate("MainWindow", "Convert"))
        self.convert_pb.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.select_acc.setItemText(0, _translate("MainWindow", "Option 1"))
        self.select_acc.setItemText(1, _translate("MainWindow", "Option 2"))
        self.convert_label.setText(_translate("MainWindow", "CONVERTED"))
        self.save_pb.setText(_translate("MainWindow", "Save Image"))
        self.save_pb.setShortcut(_translate("MainWindow", "Ctrl+S"))
