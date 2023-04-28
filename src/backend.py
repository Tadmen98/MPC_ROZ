from stl import mesh
import numpy as np
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem
from PySide6 import QtWidgets, QtCore
from threading import Thread
from src.camera import Camera
import cv2
from src.base.model_detection import Model_Detection
import json
import os
from src.base.model_registration import Model_Registration
from src.base.model_mesh import Model_Mesh
from src.base.model_points import Model_Points

def threaded(fn):
    """To use as decorator to make a function call threaded.
    Needs import
    from threading import Thread"""

    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper

class Backend(QtCore.QObject):
    update_model_signal = QtCore.Signal(MeshData)
    registration_started_signal = QtCore.Signal()
    registration_ended_signal = QtCore.Signal()

    def __init__(self):
        super(Backend, self).__init__()
        self.feature_extractors = ["ORB", "KAZE", "AKAZE", "BRISK", "SIFT"]
        self.selected_extractor = "ORB"
        self.camera_sel_cb_add_items_signal = None
        self.camera = Camera()
        self.config = {}
        self.model_registration = Model_Registration()
        self.model_mesh = Model_Mesh()
        self.model_points = None
        self.registration_started = False
        self.registration_ended = False

        self.num_keypoints = 2000
        self.ratio_test = 0.7
        self.iterations_count = 500
        self.reprojection_error = 2
        self.confidence = 0.95
        self.min_inliers = 10
        self.feature_name = "ORB"

        self.Q = None
        if os.path.exists("config.json"):
            with open("config.json","r") as file:
                try:
                    self.config = json.load(file)
                    if "Q" in self.config:
                        self.Q = cv2.Mat(np.array(self.config["Q"]['data']))
                except ValueError as e:
                    pass

        self.camera.image_update.connect(self.camera_stream_update_slot)
        

        self.model_detection_left = Model_Detection("left")
        self.model_detection_right = Model_Detection("right")
        self.camera.image_update_left.connect(self.model_detection_left.camera_slot)
        self.camera.image_update_right.connect(self.model_detection_right.camera_slot)

        

    
    def load_model_mesh(self):
        """Opens file dialog for user to select stl model.Method
        is called by Load model button.
        """

        #Open file dialog for choosing a file
        name = QtWidgets.QFileDialog.getOpenFileName() 
        if name[0] == "":
            return
        
        stl_mesh = mesh.Mesh.from_file(name[0])
        self.model_mesh.load(name[0])

        points = stl_mesh.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)

        mesh_data = MeshData(vertexes=points, faces=faces)

        self.model_detection_left.load_mesh(self.model_mesh)
        self.model_detection_right.load_mesh(self.model_mesh)
        self.update_model_signal.emit(mesh_data)
    
    def load_model_points(self): 
        #Open file dialog for choosing a file
        name = QtWidgets.QFileDialog.getOpenFileName() 
        if name[0] == "":
            return

        self.model_points = Model_Points()
        self.model_points.load(name[0])    
        
        self.model_detection_left.load_points(self.model_points)
        self.model_detection_right.load_points(self.model_points)
        self.feature_name = self.model_points.extractor
        self.update_detection_parameters()

    
    @threaded
    def find_aviable_cameras(self):
        num_of_cameras = 0
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                break
            else:
                num_of_cameras += 1
            cap.release()
        self.camera_sel_cb_add_items_signal.emit(num_of_cameras)

    @threaded
    def connect_camera(self, camera_index):
        self.camera.stop()
        while self.camera.isRunning():
            pass
            # TODO: try stop and pass old object and then create new object
        self.camera.choose_camera(camera_index)
        self.camera.start()

    def set_registration_ended(self):
        self.registration_started = False
        self.model_registration.stop()
        self.registration_ended_signal.emit()

    def register_model(self):
        if not self.registration_started:
            self.name = QtWidgets.QFileDialog.getSaveFileName(None, "Select YML configuration save location", "model.yml", "YML (*.yml)")
            if self.name[0] == "":
                return
            self.registration_started_signal.emit()
            self.registration_started = True
            self.name = self.name[0].rstrip(".yml")
            self.iteration = 0
        
        self.model_registration.set_parameters(self.name + "_" + str(self.iteration) + ".yml", 2000, self.selected_extractor)
        self.model_registration.start(self.img_left, self.model_mesh)
        self.iteration += 1
            


    def disconnect_camera(self):
        self.camera.stop()

    def calibrate(self):
        if self.camera.capture is None:
            return (False, None)
        
        #images = glob.glob('./images/*.jpg')
        #img1 = cv2.imread(images[0])
        #img2 = cv2.imread(images[1])
        img1 = self.img_left
        img2 = self.img_right
        
        CHECKERBOARD = (6,7)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objpoints = []
        imgpoints1 = [] 
        imgpoints2 = [] 
        
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret1 == True:
            objpoints.append(objp)
            corners21 = cv2.cornerSubPix(gray1, corners1, (11,11),(-1,-1), criteria)
            imgpoints1.append(corners21)
        if ret2 == True:
            corners22 = cv2.cornerSubPix(gray2, corners2, (11,11),(-1,-1), criteria)
            imgpoints2.append(corners22)
        
        m1 = None
        m2 = None
        coef1 = None
        coef2 = None
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints1,imgpoints2, m1, coef1, m2, coef2, gray1.shape[::-1], None, None)

        self.detector.cm1 = cameraMatrix1
        self.detector.cm2 = cameraMatrix2
        
        self.detector.dc1 = distCoeffs1
        self.detector.dc2 = distCoeffs2
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray1.shape[::1], R, T)
        print("Q")
        print(Q)   
        Q_array = np.array(Q)

        # Convert the numpy array to a JSON serializable object
        Q_array = {'data': Q_array.tolist()}
        self.config["Q"] = Q_array
        self.Q = Q
        self.save_config() 

        return(True, self.Q)   

    def camera_stream_update_slot(self, img1, img2):
        self.img_left = img1
        self.img_right = img2

    def update_detection_parameters(self, num_keypoints=None, ratio_test=None, iterations_count=None, 
                                    reprojection_error=None, confidence=None, min_inliers=None):
        if num_keypoints is not None:
            self.num_keypoints = num_keypoints 
        if ratio_test is not None:
            self.ratio_test = ratio_test
        if iterations_count is not None:
            self.iterations_count = iterations_count
        if reprojection_error is not None :
            self.reprojection_error = reprojection_error
        if confidence is not None:
            self.confidence = confidence
        if min_inliers is not None:
            self.min_inliers = min_inliers

        self.model_detection_left.update_parameters(self.num_keypoints, self.ratio_test, self.iterations_count, 
                                self.reprojection_error, self.confidence, self.min_inliers, self.feature_name)
        self.model_detection_right.update_parameters(self.num_keypoints, self.ratio_test, self.iterations_count, 
                                self.reprojection_error, self.confidence, self.min_inliers, self.feature_name)
        
    def set_extractor(self, name):
        self.selected_extractor = name
