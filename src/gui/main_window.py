from PySide6.QtWidgets import QMainWindow
from PySide6 import QtWidgets, QtCore, QtGui
from src.backend import Backend
import numpy as np
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem
from stl import mesh

class MainWindow(QMainWindow):

    def __init__(self, backend):
        super().__init__()

        self.backend = backend

        self.setup_ui()

    def setup_ui(self):
        pass
        self.retranslate_ui()
        self.connect_signals()
        
    def setup_ui(self):
        if not self.objectName():
            self.setObjectName(u"MainWindow")
        self.resize(774, 464)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")

        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")

        self.basic_preview_label = QtWidgets.QLabel(self.frame)
        self.basic_preview_label.setObjectName(u"basic_preview_label")

        self.horizontalLayout_2.addWidget(self.basic_preview_label)

        self.viewer_3D = GLViewWidget()
        self.horizontalLayout_2.addWidget(self.viewer_3D)

        self.augumented_preview_label = QtWidgets.QLabel(self.frame)
        self.augumented_preview_label.setObjectName(u"augumented_preview_label")

        self.horizontalLayout_2.addWidget(self.augumented_preview_label)


        self.verticalLayout.addWidget(self.frame)

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

        self.gridLayout.addWidget(self.load_btn, 2, 2, 1, 1)

        self.disconnect_btn = QtWidgets.QPushButton(self.frame_2)
        self.disconnect_btn.setObjectName(u"disconnect_btn")

        self.gridLayout.addWidget(self.disconnect_btn, 2, 0, 1, 1)

        self.connect_btn = QtWidgets.QPushButton(self.frame_2)
        self.connect_btn.setObjectName(u"connect_btn")

        self.gridLayout.addWidget(self.connect_btn, 1, 0, 1, 1)

        self.camera_select_cb = QtWidgets.QComboBox(self.frame_2)
        self.camera_select_cb.setObjectName(u"camera_select_cb")

        self.gridLayout.addWidget(self.camera_select_cb, 0, 0, 1, 1)

        self.method_cb = QtWidgets.QComboBox(self.frame_2)
        self.method_cb.setObjectName(u"method_cb")

        self.gridLayout.addWidget(self.method_cb, 0, 2, 1, 1)


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
    # setupUi

    def retranslate_ui(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("MainWindow", u"MPC-ROZ Projekt", None))
        self.basic_preview_label.setText(QtCore.QCoreApplication.translate("MainWindow", u"Preview basic", None))
        self.augumented_preview_label.setText(QtCore.QCoreApplication.translate("MainWindow", u"Augumented view", None))
        self.load_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Load Model", None))
        self.disconnect_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Disconnect Camera", None))
        self.connect_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Connect Camera", None))
    # retranslateUi

    def connect_signals(self):
        # self.camera_select_cb.activated.connect(self.camera_select_cb_action)
        self.find_all_cameras_btn.clicked.connect(self.backend.find_aviable_cameras)
        self.connect_btn.clicked.connect(lambda: self.backend.connect_camera(
            self.camera_select_cb.currentIndex()
        ))
        self.disconnect_btn.clicked.connect(self.backend.disconnect_camera)
        self.camera_sel_cb_add_items_signal.connect(self.camera_select_cb_add_items)
        
        self.backend.update_model_signal.connect(self.update_model_slot)
        self.load_btn.clicked.connect(self.backend.get_model)


    def init_backend(self):
        self.backend.camera_sel_cb_add_items_signal = self.camera_sel_cb_add_items_signal
        self.backend.camera.image_update.connect(self.camera_stream_update_slot)


    def camera_select_cb_add_items(self, num_of_cameras: int):
        self.camera_select_cb.clear()
        self.camera_select_cb.addItems([str(item) for item in range(num_of_cameras)])

    def camera_stream_update_slot(self, img1, img2):
        image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        flipped_image = cv2.flip(image, 1)
        qt_img = QImage(flipped_image.data, flipped_image.shape[1], flipped_image.shape[0],
                                   QImage.Format_RGB888)
        scaled_qt_img = qt_img.scaled(320, 240, Qt.KeepAspectRatio)
        self.basic_preview_label.setPixmap(QPixmap.fromImage(scaled_qt_img))

    def closeEvent(self, event):
        self.backend.camera.stop()

    def update_model_slot(self, mesh_data : MeshData):
        model_mesh = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        
        self.viewer_3D.clear()
        self.viewer_3D.addItem(model_mesh)

        self.viewer_3D.show()

