import cv2
import numpy as np
from PySide6 import QtCore, QtGui
from src.base.pnp_problem import PnP_Problem
from PySide6.QtGui import QMatrix4x4
from src.base.model_mesh import Model_Mesh
from src.base.model_points import Model_Points
from src.base.correspondence_matcher import Correspondence_Matcher
from src.base.draw_functions import *
from pyqtgraph import Transform3D
from threading import Thread
from scipy.spatial.transform import Rotation
from math import atan2,sqrt

def threaded(fn):
    """To use as decorator to make a function call threaded.
    Needs import
    from threading import Thread"""

    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper

class Model_Detection(QtCore.QObject):
    detection_update_signal = QtCore.Signal(cv2.Mat, str)
    pose_transformation_signal = QtCore.Signal(Transform3D)

    def __init__(self, side):
        super(Model_Detection, self).__init__()
        self.model_points_loaded = False
        self.side = side

        self.calculation_in_progress = False
        self.model_points = Model_Points()
        self.model_mesh = Model_Mesh() 
        # Intrinsic camera params from vendor
        self.camera_params = [ 385.395,   # fx
                        385.3225,  # fy
                        333.435,      # cx
                        190.276]    # cy

        self.red = (0, 0, 255)
        self.green = (0,255,0)
        self.blue = (255,0,0)
        self.yellow = (0, 255, 255)
 
        self.num_keypoints = 2000
        self.ratio_test = 0.7
        self.iterations_count = 500
        self.reprojection_error = 2
        self.confidence = 0.95
        self.min_inliers = 10
        self.pnp_method = 0
        self.feature_name = "KAZE"

        self.pnp_detection = PnP_Problem(self.camera_params)

        self.matcher = Correspondence_Matcher()  

        self.detector, self.descriptor = self.create_features(self.feature_name, self.num_keypoints)
        self.matcher.setFeatureDetector(self.detector)                                      
        self.matcher.setDescriptorExtractor(self.descriptor)                                
        self.matcher.setDescriptorMatcher(self.create_matcher(self.feature_name))        
        self.matcher.setRatio(self.ratio_test) 

    def set_camera_parameters(self, fx, fy, cx, cy):
        self.camera_params = [ fx,  
                        fy, 
                        cx,      
                        cy]    
        self.pnp_detection = PnP_Problem(self.camera_params)
    
    def update_parameters(self, num_keypoints, ratio_test, iterations_count, 
                                reprojection_error, confidence, min_inliers, feature_name):
        self.num_keypoints = num_keypoints
        self.ratio_test = ratio_test
        self.iterations_count  = iterations_count 
        self.reprojection_error = reprojection_error
        self.confidence  = confidence 
        self.min_inliers = min_inliers
        self.feature_name = feature_name

        self.detector, self.descriptor = self.create_features(self.feature_name, self.num_keypoints)
        self.matcher.setFeatureDetector(self.detector)                                      
        self.matcher.setDescriptorExtractor(self.descriptor)  
        self.matcher.setDescriptorMatcher(self.create_matcher(self.feature_name))        
        self.matcher.setRatio(self.ratio_test) 

    def load_mesh(self, model_mesh : Model_Mesh):
        self.model_mesh = model_mesh  # load an object mesh    

    def load_points(self, model_points):
        self.model_points = model_points
        self.model_points_loaded = True

    @threaded
    def camera_slot(self, img):
        if self.calculation_in_progress:
            return
        else:
            if self.model_points_loaded and self.model_mesh.loaded:
                self.calculation_in_progress = True
            else:
                return
        
        inliers_most = 0
        frame_final = img.copy()

        
        # Model info
        self.list_points3d_model = self.model_points._list_points3d_in_ 
        self.descriptors_model = self.model_points._descriptors_           
        self.keypoints_model = self.model_points._list_keypoints_

        good_measurement = False

        frame_vis = img.copy()

        good_matches = []
        keypoints_scene = []

        good_matches, keypoints_scene = self.matcher.robustMatch(img, self.descriptors_model, self.keypoints_model)

        list_points3d_model_match = [] # 3D coordinates found in the scene
        list_points2d_scene_match = [] # 2D coordinates found in the scene

        for match_index in range(len(good_matches)):
            point3d_model = self.list_points3d_model[ good_matches[match_index].trainIdx ]  # 3D point from model
            point2d_scene = keypoints_scene[ good_matches[match_index].queryIdx ].pt # 2D point from the scene
            list_points3d_model_match.append(point3d_model)         # add 3D point
            list_points2d_scene_match.append(point2d_scene)         # add 2D point

        # Draw outliers
        draw2DPoints(frame_vis, list_points2d_scene_match, self.red)

        inliers_idx = np.array([])
        list_points2d_inliers = []

        good_measurement = False
        
        if(len(good_matches) >= 4): # solvePnPRANSAC mini 4 points
            inliers_idx = self.pnp_detection.estimatePoseRANSAC( list_points3d_model_match, list_points2d_scene_match,
                                            self.pnp_method, inliers_idx,
                                            self.iterations_count, self.reprojection_error, self.confidence )



            if inliers_idx is None:
                inliers_idx = np.array([])
            
            if len(inliers_idx) < inliers_most:
                return
            else:
                inliers_most = len(inliers_idx)

            for inliers_index in range(len(inliers_idx)):
                n = inliers_idx[inliers_index][0]         # i-inlier
                point2d = list_points2d_scene_match[n]     # i-inlier point 2D
                list_points2d_inliers.append(point2d)           # add i-inlier to list

            draw2DPoints(frame_vis, list_points2d_inliers, self.blue)

            if( len(inliers_idx) >= self.min_inliers):
                good_measurement = True
        
        l = 5
        pose_points2d = []

        if(good_measurement):
            drawObjectMesh(frame_vis, self.model_mesh, self.pnp_detection, self.green)  # draw current pose

            pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([0,0,0], dtype=np.float64)))  # axis center
            pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([l,0,0], dtype=np.float64)))  # axis x
            pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([0,l,0], dtype=np.float64)))  # axis y
            pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([0,0,l], dtype=np.float64)))  # axis z

            P_matrix = QtGui.QMatrix4x4([self.pnp_detection._P_matrix_[0,0],self.pnp_detection._P_matrix_[0,1],self.pnp_detection._P_matrix_[0,2],self.pnp_detection._P_matrix_[0,3],
                                            self.pnp_detection._P_matrix_[1,0],self.pnp_detection._P_matrix_[1,1],self.pnp_detection._P_matrix_[1,2],self.pnp_detection._P_matrix_[1,3],
                                            self.pnp_detection._P_matrix_[2,0],self.pnp_detection._P_matrix_[2,1],self.pnp_detection._P_matrix_[2,2],self.pnp_detection._P_matrix_[2,3],
                                            0,0,0,1])
            
            R_matrix = self.pnp_detection._R_matrix_
            t_matrix = self.pnp_detection._t_matrix_

            mat = QMatrix4x4(R_matrix[0,0],R_matrix[0,1],R_matrix[0,2],t_matrix[0,0],
                             R_matrix[1,0],R_matrix[1,1],R_matrix[1,2],t_matrix[1,0],
                             R_matrix[2,0],R_matrix[2,1],R_matrix[2,2],t_matrix[2,0],
                             0,0,0,1)
            
            transformation = Transform3D(mat)
            self.pose_transformation_signal.emit(transformation)
        
    #TODO Tady to chce else větev, která vykreslí starou pózu pokud nebyla detekovaná nová, ale musí to být omezené na pár snímků - jen pro zvýšení robustnosti
        frame_final = frame_vis.copy()

        self.detection_update_signal.emit(frame_final, self.side)
        self.calculation_in_progress = False


    def create_features(self, feature_name, num_keypoints):
        detector = None
        descriptor = None
        if (feature_name == "ORB"):
            detector = cv2.ORB_create(num_keypoints)
            descriptor = cv2.ORB_create(num_keypoints)
        elif (feature_name == "KAZE"):
            detector = cv2.KAZE_create()
            descriptor = cv2.KAZE_create()
        elif (feature_name == "AKAZE"):
            detector = cv2.AKAZE_create()
            descriptor = cv2.AKAZE_create()
        elif (feature_name == "BRISK"):
            detector = cv2.BRISK_create()
            descriptor = cv2.BRISK_create()
        elif (feature_name == "SIFT"):
            detector = cv2.SIFT_create()
            descriptor = cv2.SIFT_create()

        return detector, descriptor

    def create_matcher(self, feature_name):
        if (feature_name == "ORB" or feature_name == "BRISK" or feature_name == "AKAZE" or feature_name == "BINBOOST"):
            return cv2.DescriptorMatcher_create("BruteForce-Hamming")
        else:
            return cv2.DescriptorMatcher_create("BruteForce")
