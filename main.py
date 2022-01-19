# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from PyQt5 import uic
from PyQt5.QtWidgets import *
from ui_window import Ui_window
from PyQt5.QtGui import *
import threading
import cv2
from skimage import io
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sklearn
from myPCA import datainput,FPCA
import math

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象

        self.ui = Ui_window()
        self.ui.setupUi(self)  #画出窗口
        self.ui.shot.clicked.connect(self.takephoto) #拍照识别
        self.ui.det.clicked.connect(self.play_video) #人脸识别
        self.ui.updata.clicked.connect(self.getimage)  # 上传图片
        self.ui.datadet.clicked.connect(self.getFiles) #检索
        self.thread = None  # 创建线程
        self.pic = np.zeros(10)

        # self.detect()

    def play_video(self):  #创建线程不断调用摄像头捕捉画面
        self.thread = threading.Thread(target=self.detect)  #线程调用的是detect
        self.thread.start()
        print("线程开启")

    def takephoto(self):
        self.cap.release()  #拍照的话就要关掉视频流

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()  # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
        # frame = frame[150:70, 426:406]
        img = np.zeros((92, 112))
        img=frame

        img=img[70:406,150:426]
        print(img)

        # kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)  #滤波没用
        # kernel /= math.sqrt((kernel * kernel).sum())
        # filtered = cv2.filter2D(img, -1, kernel)


        # cv2.imshow("M outh1", filtered)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        convert_to_qt_format = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                                          QImage.Format_RGB888)

        self.ui.show.setPixmap(QPixmap.fromImage(convert_to_qt_format)) #和视频一样的转换格式后显示
        # print("1")
        # x, y = frame.shape[0:2]
        # # frame = cv2.resize(frame, (int(y / 2), int(x / 2)))
        # print("2")
        # frame = frame[150:70, 426:406]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)  #灰度化
        data = cv2.resize(gray, dsize=(92, 112),  interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("capture.jpg", data)  # 写入图片
        self.cap.release()  # 释放


        pic=io.imread("capture.jpg")
        print(pic)
        csim, fsim=FPCA(pic)
        if len(csim)>0 :
            self.ui.label.setText("是319成员")
        else:
            self.ui.label.setText("不是319成员")

        # self.thread.join()
        # self.ui.show.setPixmap(QPixmap("capture.jpg"))

    def detect(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 拿到摄像头的流
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            cv2.rectangle(frame, (int(150), int(70)), (int(426), int(406)), (255, 255, 255), 4)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #转颜色
            convert_to_qt_format = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                                          QImage.Format_RGB888)  #转成qt格式！！！！！

            self.ui.show.setPixmap(QPixmap.fromImage(convert_to_qt_format))  #显示
        # self.cap.release()
        cv2.destroyAllWindows()

    def getimage(self):
        # 从C盘打开文件格式（*.jpg *.gif *.png *.jpeg）文件，返回路径
        image_file, _ = QFileDialog.getOpenFileName(self, 'Open file', 'FaceDB_orl',
                                                    'Image files (*.jpg *.gif *.png *.jpeg)')
        print(image_file)
        self.pic = io.imread(image_file)
        self.ui.uploadimg.setPixmap(QPixmap(image_file))

    def getFiles(self):
        # 实例化QFileDialog
        # print("5555")
        # print(self.pic)
        csim, fsim = FPCA(self.pic)
        # print(csim)
        path = 'FaceDB_orl'
        path_list = os.listdir(path)
        i = 0
        for item in csim:
            F = "%03d" % item
            print(F)
            str = os.path.join(path, F)
            P = "%02d" % fsim[i]
            print(P)
            str = os.path.join(str, P)
            str = str + '.png'
            print(str)
            i = i + 1
            if i == 1:
                print(i)
                self.ui.re1.setPixmap(QPixmap(str))
            if i == 2:
                print(i)
                self.ui.re2.setPixmap(QPixmap(str))
            if i == 3:
                print(i)
                self.ui.re3.setPixmap(QPixmap(str))
            if i == 4:
                print(i)
                self.ui.re4.setPixmap(QPixmap(str))
            if i == 5:
                print(i)
                self.ui.re5.setPixmap(QPixmap(str))
                break
            if i == 6:
                print(i)
                self.ui.re6.setPixmap(QPixmap(str))
                break












app = QApplication([])
mainwindow = MainWindow()
mainwindow.show()
app.exec_()
