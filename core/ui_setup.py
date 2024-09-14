from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, QMenuBar,
    QSizePolicy, QStatusBar, QWidget)
from PySide6.QtCore import QTimer
import cv2
from core.camera import camera # 摄像头操作相关函数
from processing import (detector, tracker, algorithm, save_video)
from processing.algorithm import algo_switch
class Ui_MainWindow(object):
    def __init__(self):
        self.centralwidget = None
        self.label = None
        #self.label_2 = None
        self.pushButton1 = None
        # 定时器用于更新界面显示
        self.timer = QTimer()


    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(2000, 1200)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        # label
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(200, 40, 680, 480))
        # label_2
        #self.label_2 = QLabel(self.centralwidget)
        #self.label_2.setObjectName(u"label_2")
        #self.label_2.setGeometry(QRect(1000, 40, 960, 540))
        # 开始和暂停按钮

        # pushButton1_1
        self.pushButton11 = QPushButton(self.centralwidget)
        self.pushButton11.setObjectName(u"pushButton11")
        self.pushButton11.setGeometry(QRect(200, 600, 120, 100))
        self.pushButton11.setText("开始监控")
        self.pushButton11.clicked.connect(self.startMonitoring)
        # pushButton1_2
        self.pushButton12 = QPushButton(self.centralwidget)
        self.pushButton12.setObjectName(u"pushButton12")
        self.pushButton12.setGeometry(QRect(200, 800, 120, 100))
        self.pushButton12.setText("暂停监控")
        self.pushButton12.clicked.connect(self.stopMonitoring)

        # 预处理算法按钮
        # pushButton2_1
        self.pushButton21 = QPushButton(self.centralwidget)
        self.pushButton21.setObjectName(u"pushButton21")
        self.pushButton21.setGeometry(QRect(400, 600, 120, 100))
        self.pushButton21.setText("高斯滤波")
        self.pushButton21.clicked.connect(lambda:algo_switch.switch_algorithm("gaussFilter"))

        # pushButton2_2
        self.pushButton22 = QPushButton(self.centralwidget)
        self.pushButton22.setObjectName(u"pushButton22")
        self.pushButton22.setGeometry(QRect(400, 800, 120, 100))
        self.pushButton22.setText("双边滤波")
        self.pushButton22.clicked.connect(lambda:algo_switch.switch_algorithm("bilatFilter"))

        # pushButton2_3
        self.pushButton23 = QPushButton(self.centralwidget)
        self.pushButton23.setObjectName(u"pushButton23")
        self.pushButton23.setGeometry(QRect(600, 600, 120, 100))
        self.pushButton23.setText("中值滤波")
        self.pushButton23.clicked.connect(lambda:algo_switch.switch_algorithm("meanFilter"))


        # pushButton3_1
        self.pushButton31 = QPushButton(self.centralwidget)
        self.pushButton31.setObjectName(u"pushButton31")
        self.pushButton31.setGeometry(QRect(600, 800, 120, 100))
        self.pushButton31.setText("浮点灰度处理")
        self.pushButton31.clicked.connect(lambda:algo_switch.switch_algorithm("floatGray"))


        # pushButton3_2
        self.pushButton32 = QPushButton(self.centralwidget)
        self.pushButton32.setObjectName(u"pushButton32")
        self.pushButton32.setGeometry(QRect(800, 600, 120, 100))
        self.pushButton32.setText("整型灰度处理")
        self.pushButton32.clicked.connect(lambda:algo_switch.switch_algorithm("intGray"))

        # pushButton3_3
        self.pushButton33 = QPushButton(self.centralwidget)
        self.pushButton33.setObjectName(u"pushButton33")
        self.pushButton33.setGeometry(QRect(800, 800, 120, 100))
        self.pushButton33.setText("移位灰度处理")
        self.pushButton33.clicked.connect(lambda:algo_switch.switch_algorithm("moveGray"))

        # pushButton4_1
        self.pushButton41 = QPushButton(self.centralwidget)
        self.pushButton41.setObjectName(u"pushButton41")
        self.pushButton41.setGeometry(QRect(1000, 600, 120, 100))
        self.pushButton41.setText("切换质心追踪算法")
        self.pushButton41.clicked.connect(lambda:algo_switch.switch_algorithm("centroid"))

        # pushButton4_2
        self.pushButton42 = QPushButton(self.centralwidget)
        self.pushButton42.setObjectName(u"pushButton42")
        self.pushButton42.setGeometry(QRect(1000, 800, 120, 100))
        self.pushButton42.setText("保存视频")
        self.pushButton42.clicked.connect(lambda:save_video.video_record())

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"跨场景跟踪系统", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"录像暂停中", None))
        #self.label_2.setText(QCoreApplication.translate("MainWindow", u"录像暂停中", None))
    # retranslateUi

    
    
    # 退出程序结束监控
    def stopMonitoring(self):
        self.timer.stop() # 停止计时器
        #self.capture2.release() # 释放摄像头对象
        self.capture.release()
        print("程序已结束！")

    
    
    # 开启摄像头开始监控
    def startMonitoring(self):
        self.timer.timeout.connect(self.turnOnCamera)
        self.timer.start(50) 

   

     # 打开摄像头的函数
    def turnOnCamera(self):
        # 开启摄像头读取视频帧，并将其更新到QLabel上
        # 读取每帧图片
            _,im = camera.capture.read()
            #_,im2 = camera.capture2.read()
            #缩小尺寸，1920x1080->960x540
            im = cv2.resize(im, (640,480))
            #im2 = cv2.resize(im2, (960, 540))
            list_bboxs = []
            bboxes = detector.Detector.detect(im)
            # 如果画面中有bbox
            if len(bboxes) > 0:
                list_bboxs = tracker.update(bboxes, im)
                # 画框
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
                pass
            else:
                # 如果画面中 没有bbox
                output_image_frame = im
            pass
    
            # 调整 output_image_frame 的尺寸与 camera.color_polygons_image 一致
            output_image_frame = cv2.resize(output_image_frame, (camera.color_polygons_image.shape[1], camera.color_polygons_image.shape[0]))

            # # 执行图像相加操作
            output_image_frame = cv2.add(output_image_frame, camera.color_polygons_image)
            if len(list_bboxs) > 0:
                # ----------------------判断撞线----------------------
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, _, track_id = item_bbox

                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))

                    # 撞线的点
                    y = y1_offset
                    x = x1

                    if camera.polygon_mask_blue_and_yellow[y, x] == 1:
                        # 如果撞 蓝polygon
                        if track_id not in camera.list_overlapping_blue_polygon:
                            camera.list_overlapping_blue_polygon.append(track_id)
                        pass

                        # 判断 黄polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 外出方向
                        if track_id in camera.list_overlapping_yellow_polygon:
                            # 外出+1
                            camera.up_count += 1

                            print('up count:', camera.up_count, ', up id:', camera.list_overlapping_yellow_polygon)

                            # 删除 黄polygon list 中的此id
                            camera.list_overlapping_yellow_polygon.remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass

                    elif camera.polygon_mask_blue_and_yellow[y, x] == 2:
                        # 如果撞 黄polygon
                        if track_id not in camera.list_overlapping_yellow_polygon:
                            camera.list_overlapping_yellow_polygon.append(track_id)
                        pass

                        # 判断 蓝polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 进入方向
                        if track_id in camera.list_overlapping_blue_polygon:
                            # 进入+1
                            camera.down_count += 1
                            print('down count:', camera.down_count, ', down id:', camera.list_overlapping_blue_polygon)
                            # 删除 蓝polygon list 中的此id
                            camera.list_overlapping_blue_polygon.remove(track_id)
                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass
                        pass
                    else:
                        pass
                    pass

                pass

            # ----------------------清除无用id----------------------
                list_overlapping_all = camera.list_overlapping_yellow_polygon + camera.list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                            break
                        pass
                    pass

                    if not is_found:
                        # 如果没找到，删除id
                        if id1 in camera.list_overlapping_yellow_polygon:
                            camera.list_overlapping_yellow_polygon.remove(id1)
                        pass
                        if id1 in camera.list_overlapping_blue_polygon:
                            camera.list_overlapping_blue_polygon.remove(id1)
                        pass
                    pass
                list_overlapping_all.clear()
                pass

                # 清空list
                list_bboxs.clear()

                pass
            else:
                # 如果图像中没有任何的bbox，则清空list
                camera.list_overlapping_blue_polygon.clear()
                camera.list_overlapping_yellow_polygon.clear()
                pass
            pass

            text_draw = 'DOWN: ' + str(camera.down_count) + \
                    ' , UP: ' + str(camera.up_count)
            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                            org=camera.draw_text_postion,
                                            fontFace=camera.font_draw_number,
                                            fontScale=1, color=(255, 255, 255), thickness=2)
            camera.tmp_im = output_image_frame
            output_image_frame = cv2.cvtColor(output_image_frame, cv2.COLOR_BGR2RGB)
            im = algorithm.algo_switch.check_active_algo(output_image_frame)
            #camera.tmp_im2 = im
            #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            self.pixmap1 = QImage(im, 640, 480, QImage.Format_RGB888)
            self.pixmap1 = QPixmap.fromImage(self.pixmap1)
            #self.pixmap2 = QImage(im2, 960, 540, QImage.Format_RGB888)
            #self.pixmap2 = QPixmap.fromImage(self.pixmap2)
            self.label.setPixmap(self.pixmap1)
            #self.label_2.setPixmap(self.pixmap2)