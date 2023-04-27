from PySide6 import QtWidgets, QtCore, QtGui
from src.backend import Backend
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Signal, Slot, Qt
import numpy as np
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem,GLScatterPlotItem
from stl import mesh
import cv2
from src.backend import Backend

class MainWindow(QtWidgets.QMainWindow):
    
    camera_sel_cb_add_items_signal = Signal(int)

    def __init__(self, backend: Backend):
        super().__init__()

        self.backend = backend

        self.img_not_connected = QtGui.QImage("images/not_connected.png")

        self.setup_ui()
        self.retranslate_ui()
        self.init_backend()
        self.connect_signals()
        self.camera_disconnected()
        
    def setup_ui(self):
        if not self.objectName():
            self.setObjectName(u"MainWindow")
        self.resize(774, 464)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")

        #LEFT CAM
        self.frame_left = QtWidgets.QFrame(self.centralwidget)
        self.frame_left.setObjectName(u"frame_left")
        self.frame_left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalLayout_left = QtWidgets.QHBoxLayout(self.frame_left)
        self.horizontalLayout_left.setObjectName(u"horizontalLayout_left")

        self.basic_preview_label_left = QtWidgets.QLabel(self.frame_left)
        self.basic_preview_label_left.setObjectName(u"basic_preview_label_left")

        self.horizontalLayout_left.addWidget(self.basic_preview_label_left)

        self.viewer_3D_left = GLViewWidget()
        self.horizontalLayout_left.addWidget(self.viewer_3D_left)

        self.augumented_preview_label_left = QtWidgets.QLabel(self.frame_left)
        self.augumented_preview_label_left.setObjectName(u"augumented_preview_label_left")

        self.horizontalLayout_left.addWidget(self.augumented_preview_label_left)
        self.verticalLayout.addWidget(self.frame_left)

        #RIGHT CAM
        self.frame_right = QtWidgets.QFrame(self.centralwidget)
        self.frame_right.setObjectName(u"frame_right")
        self.frame_right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_right.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalLayout_right = QtWidgets.QHBoxLayout(self.frame_right)
        self.horizontalLayout_right.setObjectName(u"horizontalLayout_right")

        self.basic_preview_label_right = QtWidgets.QLabel(self.frame_right)
        self.basic_preview_label_right.setObjectName(u"basic_preview_label_right")

        self.horizontalLayout_right.addWidget(self.basic_preview_label_right)

        self.viewer_3D_right = GLViewWidget()
        self.horizontalLayout_right.addWidget(self.viewer_3D_right)

        self.augumented_preview_label_right = QtWidgets.QLabel(self.frame_right)
        self.augumented_preview_label_right.setObjectName(u"augumented_preview_label_right")

        self.horizontalLayout_right.addWidget(self.augumented_preview_label_right)
        self.verticalLayout.addWidget(self.frame_right)
        ###--------------------

        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())

        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)

        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setObjectName(u"gridLayout")

        self.load_btn = QtWidgets.QPushButton(self.frame_2)
        self.load_btn.setObjectName(u"load_btn")

        self.gridLayout.addWidget(self.load_btn, 2, 2, 1, 2)

        self.calibrate_btn = QtWidgets.QPushButton(self.frame_2)
        self.calibrate_btn.setObjectName(u"calibrate_btn")

        self.gridLayout.addWidget(self.calibrate_btn, 1, 2, 1, 2)

        self.disconnect_btn = QtWidgets.QPushButton(self.frame_2)
        self.disconnect_btn.setObjectName(u"disconnect_btn")

        self.gridLayout.addWidget(self.disconnect_btn, 2, 0, 1, 2)

        self.connect_btn = QtWidgets.QPushButton(self.frame_2)
        self.connect_btn.setObjectName(u"connect_btn")

        self.gridLayout.addWidget(self.connect_btn, 1, 0, 1, 2)

        self.register_model_btn = QtWidgets.QPushButton(self.frame_2)
        self.register_model_btn.setObjectName(u"register_model_btn")

        self.gridLayout.addWidget(self.register_model_btn, 3, 2, 1, 2)

        self.camera_select_cb = QtWidgets.QComboBox(self.frame_2)
        self.camera_select_cb.setObjectName(u"camera_select_cb")

        self.gridLayout.addWidget(self.camera_select_cb, 0, 0, 1, 1)

        self.find_all_cameras_btn = QtWidgets.QPushButton(self.frame_2)
        self.find_all_cameras_btn.setObjectName(u"find_all_cameras_btn")

        self.gridLayout.addWidget(self.find_all_cameras_btn, 0, 1, 1, 1)

        self.method_cb = QtWidgets.QComboBox(self.frame_2)
        self.method_cb.setObjectName(u"method_cb")

        self.gridLayout.addWidget(self.method_cb, 0, 2, 1, 2)

        self.DEBUG_btn = QtWidgets.QPushButton(self.frame_2)
        self.DEBUG_btn.setObjectName(u"DEBUG_btn")

        self.gridLayout.addWidget(self.DEBUG_btn, 1, 0, 3, 2)

        self.verticalLayout.addWidget(self.frame_2)

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 774, 21))
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)

        QtCore.QMetaObject.connectSlotsByName(self)
   

    def retranslate_ui(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("MainWindow", u"MPC-ROZ Projekt", None))
        self.load_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Load Model", None))
        self.calibrate_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Calibrate Camera", None))
        self.register_model_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Register Model", None))
        self.DEBUG_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"DEBUG", None))
        self.disconnect_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Disconnect Camera", None))
        self.connect_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Connect Camera", None))
        self.find_all_cameras_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Find all cameras", None))

        

    def connect_signals(self):
        self.DEBUG_btn.clicked.connect(self.backend.set_registration_ended)
        self.register_model_btn.clicked.connect(self.backend.register_model)
        self.backend.update_model_signal.connect(self.update_model_slot)
        #self.backend.detector.update_scene_signal.connect(self.update_scene_slot)
        self.load_btn.clicked.connect(self.backend.get_model)
        self.calibrate_btn.clicked.connect(self.backend.calibrate)
        self.find_all_cameras_btn.clicked.connect(self.backend.find_aviable_cameras)
        self.connect_btn.clicked.connect(lambda: self.backend.connect_camera(
            self.camera_select_cb.currentIndex()
        ))
        self.disconnect_btn.clicked.connect(self.backend.disconnect_camera)
        self.camera_sel_cb_add_items_signal.connect(self.camera_select_cb_add_items)


    def update_model_slot(self, mesh_data : MeshData):
        model_mesh = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        
        self.viewer_3D_left.clear()
        self.viewer_3D_left.addItem(model_mesh)
        self.viewer_3D_left.show()

        self.viewer_3D_right.clear()
        self.viewer_3D_right.addItem(model_mesh)
        self.viewer_3D_right.show()

    def init_backend(self):
        self.backend.camera_sel_cb_add_items_signal = self.camera_sel_cb_add_items_signal
        self.backend.camera.image_update.connect(self.camera_stream_update_slot)
        self.backend.model_detection_left.detection_update_signal.connect(self.detection_update_slot)
        self.backend.model_detection_right.detection_update_signal.connect(self.detection_update_slot)


    def camera_select_cb_add_items(self, num_of_cameras: int):
        self.camera_select_cb.clear()
        self.camera_select_cb.addItems([str(item) for item in range(num_of_cameras)])

    def camera_stream_update_slot(self, img_left, img_right):
        image_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        
        qt_img_left = QImage(image_left.data, image_left.shape[1], image_left.shape[0],
                                   QImage.Format_RGB888)
        scaled_qt_img_left = qt_img_left.scaledToWidth(self.width()/3)
        self.basic_preview_label_left.setPixmap(QPixmap.fromImage(scaled_qt_img_left))

        #____RIGHT___#
        image_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
        
        qt_img = QImage(image_right.data, image_right.shape[1], image_right.shape[0],
                                   QImage.Format_RGB888)
        scaled_qt_img_right = qt_img.scaledToWidth(self.width()/3)
        self.basic_preview_label_right.setPixmap(QPixmap.fromImage(scaled_qt_img_right))

    def detection_update_slot(self, img, side):
        if side == "left":
            image_left = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_img_left = QImage(image_left.data, image_left.shape[1], image_left.shape[0],
                                    QImage.Format_RGB888)
            scaled_qt_img_left = qt_img_left.scaledToWidth(self.width()/3)
            self.augumented_preview_label_left.setPixmap(QPixmap.fromImage(scaled_qt_img_left))
        elif side == "right":
            #____RIGHT___#
            image_right = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_img_right = QImage(image_right.data, image_right.shape[1], image_right.shape[0],
                                    QImage.Format_RGB888)
            scaled_qt_img_right = qt_img_right.scaledToWidth(self.width()/3)
            self.augumented_preview_label_right.setPixmap(QPixmap.fromImage(scaled_qt_img_right))

    def closeEvent(self, event):
        self.backend.camera.stop()

    def camera_disconnected(self):
        scaled_qt_img = self.img_not_connected.scaledToWidth(self.width()/3)
        self.basic_preview_label_left.setPixmap(QPixmap.fromImage(scaled_qt_img))
        self.augumented_preview_label_left.setPixmap(QPixmap.fromImage(scaled_qt_img))
        self.basic_preview_label_right.setPixmap(QPixmap.fromImage(scaled_qt_img))
        self.augumented_preview_label_right.setPixmap(QPixmap.fromImage(scaled_qt_img))
