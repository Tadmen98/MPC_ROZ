from PySide6 import QtWidgets, QtCore, QtGui
from src.backend import Backend
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Signal, Slot, Qt
import numpy as np
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem,GLScatterPlotItem
from pyqtgraph import Vector
from stl import mesh
import cv2
from src.backend import Backend
from math import atan2, sqrt
from scipy.spatial.transform import Rotation as R

class MainWindow(QtWidgets.QMainWindow):
    
    camera_sel_cb_add_items_signal = Signal(int)

    def __init__(self, backend: Backend):
        super().__init__()

        self.backend = backend
        self.translate_old = (0,0,0)

        self.img_not_connected = QtGui.QImage("images/not_connected.png")
        self.model_mesh_left = GLMeshItem()
        self.model_mesh_right = GLMeshItem()

        self.setup_ui()
        self.retranslate_ui()
        self.init_backend()
        self.connect_signals()
        self.camera_disconnected()

        self.point_mesh = None
        
        self.backend.model_detection_left.pose_transformation_signal.connect(lambda trans: self.transform_3d_view(trans, "left"))
        self.backend.model_detection_right.pose_transformation_signal.connect(lambda trans: self.transform_3d_view(trans, "right"))
        
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

        self.frame_loading = QtWidgets.QFrame(self.centralwidget)
        self.frame_loading.setObjectName(u"frame_loading")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_loading.sizePolicy().hasHeightForWidth())

        self.frame_loading.setSizePolicy(sizePolicy)
        self.frame_loading.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_loading.setFrameShadow(QtWidgets.QFrame.Raised)

        self.gridLayout = QtWidgets.QGridLayout(self.frame_loading)
        self.gridLayout.setObjectName(u"gridLayout")

        self.load_mesh_btn = QtWidgets.QPushButton(self.frame_loading)
        self.load_mesh_btn.setObjectName(u"load_mesh_btn")

        self.load_points_btn = QtWidgets.QPushButton(self.frame_loading)
        self.load_points_btn.setObjectName(u"load_points_btn")


        #self.calibrate_btn = QtWidgets.QPushButton(self.frame_loading)
        #self.calibrate_btn.setObjectName(u"calibrate_btn")

        #self.gridLayout.addWidget(self.calibrate_btn, 1, 2, 1, 2)

        #self.disconnect_btn = QtWidgets.QPushButton(self.frame_loading)
        #self.disconnect_btn.setObjectName(u"disconnect_btn")

        #self.gridLayout.addWidget(self.disconnect_btn, 2, 0, 1, 2)

        self.connect_btn = QtWidgets.QPushButton(self.frame_loading)
        self.connect_btn.setObjectName(u"connect_btn")


        self.register_model_btn = QtWidgets.QPushButton(self.frame_loading)
        self.register_model_btn.setObjectName(u"register_model_btn")
        self.register_model_btn.setDisabled(True)


        self.camera_select_cb = QtWidgets.QComboBox(self.frame_loading)
        self.camera_select_cb.setObjectName(u"camera_select_cb")


        self.find_all_cameras_btn = QtWidgets.QPushButton(self.frame_loading)
        self.find_all_cameras_btn.setObjectName(u"find_all_cameras_btn")

        self.label_extractor = QtWidgets.QLabel(self.frame_loading)
        self.label_extractor.setObjectName(u"label_extractor")

        self.finish_register_model_btn = QtWidgets.QPushButton(self.frame_loading)
        self.finish_register_model_btn.setObjectName(u"finish_register_model_btn")
        self.finish_register_model_btn.setDisabled(True)

        self.combo_box_extractor = QtWidgets.QComboBox(self.frame_loading)
        self.combo_box_extractor.setObjectName(u"combo_box_extractor")
        self.combo_box_extractor.addItems(self.backend.feature_extractors)
        self.combo_box_extractor.setCurrentIndex(0)

        self.btn_helper_scale_up = QtWidgets.QPushButton(self.frame_loading)
        self.btn_helper_scale_up.setObjectName(u"btn_helper_scale_up")
        self.btn_helper_scale_up.setText("Helper scale up")
        self.btn_helper_scale_down = QtWidgets.QPushButton(self.frame_loading)
        self.btn_helper_scale_up.setObjectName(u"btn_helper_scale_up")
        self.btn_helper_scale_down.setText("Helper scale down")

        
        self.gridLayout.addWidget(self.camera_select_cb,            0, 0, 1, 1)
        self.gridLayout.addWidget(self.find_all_cameras_btn,        0, 1, 1, 1)
        self.gridLayout.addWidget(self.connect_btn,                 1, 0, 1, 2)
        self.gridLayout.addWidget(self.label_extractor,             2, 0, 1, 2)
        self.gridLayout.addWidget(self.combo_box_extractor,         3, 0, 1, 2)

        self.gridLayout.addWidget(self.load_mesh_btn,               0, 2, 1, 2)
        self.gridLayout.addWidget(self.load_points_btn,             1, 2, 1, 2)
        self.gridLayout.addWidget(self.register_model_btn,          2, 2, 1, 2)
        self.gridLayout.addWidget(self.finish_register_model_btn,   3, 2, 1, 2)
        self.gridLayout.addWidget(self.btn_helper_scale_down,       4, 0, 1, 1)
        self.gridLayout.addWidget(self.btn_helper_scale_up,         4, 1, 1, 1)

        self.verticalLayout.addWidget(self.frame_loading)
        #__________________________________________
        self.frame_parameters = QtWidgets.QFrame(self.centralwidget)
        self.frame_parameters.setObjectName(u"frame_loading")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_parameters.sizePolicy().hasHeightForWidth())

        self.label_detection_parameters = QtWidgets.QLabel(self.frame_parameters)
        self.label_detection_parameters.setObjectName(u"label_detection_parameters")

        self.label_num_keypoints = QtWidgets.QLabel(self.frame_parameters)
        self.label_num_keypoints.setObjectName(u"label_num_keypoints")
        self.slider_num_keypoints = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.frame_parameters)
        self.slider_num_keypoints.setMinimum(5)
        self.slider_num_keypoints.setMaximum(2000)
        self.slider_num_keypoints.setValue(2000)

        self.label_ratio_test = QtWidgets.QLabel(self.frame_parameters)
        self.label_ratio_test.setObjectName(u"label_ratio_test")
        self.slider_ratio_test = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.frame_parameters)
        self.slider_ratio_test.setMinimum(0)
        self.slider_ratio_test.setMaximum(100)
        self.slider_ratio_test.setValue(0.7)

        self.label_iteration_count = QtWidgets.QLabel(self.frame_parameters)
        self.label_iteration_count.setObjectName(u"label_iteration_count")
        self.slider_iteration_count = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.frame_parameters)
        self.slider_iteration_count.setMinimum(1)
        self.slider_iteration_count.setMaximum(1000)
        self.slider_iteration_count.setValue(500)

        self.label_reprojection_error = QtWidgets.QLabel(self.frame_parameters)
        self.label_reprojection_error.setObjectName(u"label_reprojection_error")
        self.slider_reprojection_error = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.frame_parameters)
        self.slider_reprojection_error.setMinimum(0)
        self.slider_reprojection_error.setMaximum(100)
        self.slider_reprojection_error.setValue(2)

        self.label_confidence = QtWidgets.QLabel(self.frame_parameters)
        self.label_confidence.setObjectName(u"label_confidence")
        self.slider_confidence = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.frame_parameters)
        self.slider_confidence.setMinimum(1)
        self.slider_confidence.setMaximum(99)
        self.slider_confidence.setValue(95)

        self.label_min_inliers = QtWidgets.QLabel(self.frame_parameters)
        self.label_min_inliers.setObjectName(u"label_min_inliers")
        self.slider_min_inliers = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.frame_parameters)
        self.slider_min_inliers.setMinimum(1)
        self.slider_min_inliers.setMaximum(2000)
        self.slider_min_inliers.setValue(100)

        self.label_active_descriptor = QtWidgets.QLabel(self.frame_parameters)
        self.label_active_descriptor.setObjectName(u"label_active_descriptor")
        self.active_descriptor = QtWidgets.QLabel(self.frame_parameters)
        self.active_descriptor.setObjectName(u"active_descriptor")

        #---------------------

        self.value_slider_confidence = QtWidgets.QLabel(self.frame_parameters)
        self.value_slider_confidence.setObjectName(u"value_slider_confidence")

        self.value_slider_min_inliers = QtWidgets.QLabel(self.frame_parameters)
        self.value_slider_min_inliers.setObjectName(u"value_slider_confidence")

        self.value_slider_reprojection_error = QtWidgets.QLabel(self.frame_parameters)
        self.value_slider_reprojection_error.setObjectName(u"value_slider_confidence")

        self.value_slider_iteration_count = QtWidgets.QLabel(self.frame_parameters)
        self.value_slider_iteration_count.setObjectName(u"value_slider_confidence")

        self.value_slider_ratio_test = QtWidgets.QLabel(self.frame_parameters)
        self.value_slider_ratio_test.setObjectName(u"value_slider_confidence")

        self.value_slider_num_keypoints = QtWidgets.QLabel(self.frame_parameters)
        self.value_slider_num_keypoints.setObjectName(u"value_slider_confidence")

        #------------------------

        self.gridLayout_parameters = QtWidgets.QGridLayout(self.frame_parameters)
        self.gridLayout_parameters.setObjectName(u"gridLayout_parameters")
        
        self.gridLayout_parameters.addWidget(self.label_detection_parameters, 0, 0, 1, 1)

        self.gridLayout_parameters.addWidget(self.label_num_keypoints,      1, 0, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_ratio_test,         2, 0, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_iteration_count,    3, 0, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_reprojection_error, 1, 3, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_confidence,         2, 3, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_min_inliers,        3, 3, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_active_descriptor,  0, 3, 1, 1)
        self.gridLayout_parameters.addWidget(self.slider_num_keypoints,     1, 1, 1, 1)
        self.gridLayout_parameters.addWidget(self.slider_ratio_test,        2, 1, 1, 1)
        self.gridLayout_parameters.addWidget(self.slider_iteration_count,   3, 1, 1, 1)
        self.gridLayout_parameters.addWidget(self.slider_reprojection_error,1, 4, 1, 1)
        self.gridLayout_parameters.addWidget(self.slider_confidence,        2, 4, 1, 1)
        self.gridLayout_parameters.addWidget(self.slider_min_inliers,       3, 4, 1, 1)
        self.gridLayout_parameters.addWidget(self.active_descriptor,        0, 4, 1, 1)

        self.gridLayout_parameters.addWidget(self.value_slider_num_keypoints,     1, 2, 1, 1)
        self.gridLayout_parameters.addWidget(self.value_slider_ratio_test,        2, 2, 1, 1)
        self.gridLayout_parameters.addWidget(self.value_slider_iteration_count,   3, 2, 1, 1)
        self.gridLayout_parameters.addWidget(self.value_slider_reprojection_error,1, 5, 1, 1)
        self.gridLayout_parameters.addWidget(self.value_slider_confidence,        2, 5, 1, 1)
        self.gridLayout_parameters.addWidget(self.value_slider_min_inliers,       3, 5, 1, 1)

        self.verticalLayout.addWidget(self.frame_parameters)

        #__________________________________________

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
        self.load_mesh_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Load Model Mesh", None))
        self.label_extractor.setText(QtCore.QCoreApplication.translate("MainWindow", u"Feature Extractor", None))
        self.load_points_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Load Model Points", None))
        self.register_model_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Register Model", None))
        self.finish_register_model_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Finish Model Registration", None))
        self.connect_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Connect Camera", None))
        self.find_all_cameras_btn.setText(QtCore.QCoreApplication.translate("MainWindow", u"Find all cameras", None))

        self.label_num_keypoints.setText(QtCore.QCoreApplication.translate("MainWindow", u"Number of keypoints", None))
        self.label_ratio_test.setText(QtCore.QCoreApplication.translate("MainWindow", u"Ratio test", None))
        self.label_iteration_count.setText(QtCore.QCoreApplication.translate("MainWindow", u"Iteration count", None))
        self.label_reprojection_error.setText(QtCore.QCoreApplication.translate("MainWindow", u"Reprojection error", None))
        self.label_confidence.setText(QtCore.QCoreApplication.translate("MainWindow", u"Confidence", None))
        self.label_min_inliers.setText(QtCore.QCoreApplication.translate("MainWindow", u"Minimum inliers", None))
        self.label_active_descriptor.setText(QtCore.QCoreApplication.translate("MainWindow", u"Current descriptor", None))

        self.value_slider_num_keypoints.setText(str(self.slider_num_keypoints.value()))
        self.value_slider_ratio_test.setText(str(self.slider_ratio_test.value()))
        self.value_slider_iteration_count.setText(str(self.slider_iteration_count.value()))
        self.value_slider_reprojection_error.setText(str(self.slider_reprojection_error.value()))
        self.value_slider_confidence.setText(str(self.slider_confidence.value()))
        self.value_slider_min_inliers.setText(str(self.slider_min_inliers.value()))


        

    def connect_signals(self):
        self.finish_register_model_btn.clicked.connect(self.backend.set_registration_ended)
        self.register_model_btn.clicked.connect(self.backend.register_model)
        self.backend.update_model_signal.connect(self.update_model_slot)
        #self.backend.detector.update_scene_signal.connect(self.update_scene_slot)
        self.load_mesh_btn.clicked.connect(self.backend.load_model_mesh)
        self.load_points_btn.clicked.connect(self.backend.load_model_points)
        self.find_all_cameras_btn.clicked.connect(self.backend.find_aviable_cameras)
        self.connect_btn.clicked.connect(lambda: self.backend.connect_camera(
            self.camera_select_cb.currentIndex()
        ))
        self.camera_sel_cb_add_items_signal.connect(self.camera_select_cb_add_items)
        self.backend.registration_started_signal.connect(lambda: self.finish_register_model_btn.setDisabled(False))
        self.backend.registration_ended_signal.connect(lambda: self.finish_register_model_btn.setDisabled(True))
        self.backend.camera_connected_signal.connect(self.register_enabler)
        self.backend.update_model_signal.connect(self.register_enabler)

        self.combo_box_extractor.currentIndexChanged.connect(lambda: self.backend.set_extractor(self.combo_box_extractor.currentText()))
        self.backend.registration_started_signal.connect(lambda: self.combo_box_extractor.setDisabled(True))
        self.backend.registration_ended_signal.connect(lambda: self.combo_box_extractor.setDisabled(False))
        
        self.slider_num_keypoints.valueChanged.connect(lambda a: self.value_slider_num_keypoints.setText(str(a)))
        self.slider_ratio_test.valueChanged.connect(lambda a: self.value_slider_ratio_test.setText("{:.2f}".format(a/100)))    
        self.slider_iteration_count.valueChanged.connect(lambda a: self.value_slider_iteration_count.setText(str(a)))
        self.slider_reprojection_error.valueChanged.connect(lambda a: self.value_slider_reprojection_error.setText(str(a)))
        self.slider_confidence.valueChanged.connect(lambda a: self.value_slider_confidence.setText("{:.2f}".format(a/100)))
        self.slider_min_inliers.valueChanged.connect(lambda a: self.value_slider_min_inliers.setText(str(a)))

        self.slider_num_keypoints.valueChanged.connect(lambda a: self.backend.update_detection_parameters(num_keypoints = int(a)))
        self.slider_ratio_test.valueChanged.connect(lambda a: self.backend.update_detection_parameters(ratio_test = a/100))  
        self.slider_iteration_count.valueChanged.connect(lambda a: self.backend.update_detection_parameters(iterations_count=int(a)))
        self.slider_reprojection_error.valueChanged.connect(lambda a: self.backend.update_detection_parameters(reprojection_error=int(a)))
        self.slider_confidence.valueChanged.connect(lambda a: self.backend.update_detection_parameters(confidence=a/100))
        self.slider_min_inliers.valueChanged.connect(lambda a: self.backend.update_detection_parameters(min_inliers=int(a)))

        self.backend.model_registration.registration_update_signal.connect(self.show_current_point)

    def show_current_point(self, x, y, z):
        if self.point_mesh is None:
            stl_mesh = mesh.Mesh.from_file("images/point.stl")
            points = stl_mesh.points.reshape(-1, 3)
            faces = np.arange(points.shape[0]).reshape(-1, 3)
            mesh_data = MeshData(vertexes=points, faces=faces)
            self.point_mesh = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=True, edgeColor=(1, 0, 0, 1),faceColor=(1, 0, 0, 1))
            self.point_mesh.scale(0.1,0.1,0.1)
            self.viewer_3D_left.addItem(self.point_mesh)
            self.btn_helper_scale_up.clicked.connect(lambda: self.point_mesh.scale(1.5,1.5,1.5))
            self.btn_helper_scale_down.clicked.connect(lambda: self.point_mesh.scale(0.5,0.5,0.5))

        
        self.point_mesh.translate(self.translate_old[0], self.translate_old[1], self.translate_old[2])
        self.translate_old = (-x,-y,-z)
        self.point_mesh.translate(x,y,z)

        self.viewer_3D_left.show()

    def update_model_slot(self, mesh_data : MeshData):
        self.model_mesh_left = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.model_mesh_right = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        
        
        self.viewer_3D_left.clear()
        self.viewer_3D_left.addItem(self.model_mesh_left)
        self.viewer_3D_left.show()

        self.viewer_3D_right.clear()
        self.viewer_3D_right.addItem(self.model_mesh_right)
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

    def transform_3d_view(self, trans_mat : QtGui.QMatrix4x4, side):
        if side == "left":
            self.model_mesh_left.applyTransform(trans_mat,False)
        elif side == "right":
            self.model_mesh_right.applyTransform(trans_mat,False)

    def register_enabler(self):
        if self.backend.camera_connected and self.backend.mesh_loaded:
            self.register_model_btn.setDisabled(False)