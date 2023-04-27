import cv2
import numpy as np
from PySide6 import QtCore
from src.base.pnp_problem import PnP_Problem
from src.base.model_mesh import Model_Mesh
from src.base.model_points import Model_Points
from src.base.correspondence_matcher import Correspondence_Matcher
from src.base.draw_functions import *
from threading import Thread

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
    detection_update_signal = QtCore.Signal(cv2.Mat)

    def __init__(self):
        super(Model_Detection, self).__init__()
        self.model_points_loaded = False
   
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
 
        self.numKeyPoints = 2000
        self.ratioTest = 0.7
        self.iterationsCount = 500
        self.reprojectionError = 2
        self.confidence = 0.95
        self.minInliers = 10
        self.pnpMethod = 0
        self.featureName = "KAZE"

        self.pnp_detection = PnP_Problem(self.camera_params)

        self.matcher = Correspondence_Matcher()  

        self.detector, self.descriptor = self.create_features(self.featureName, self.numKeyPoints)
        self.matcher.setFeatureDetector(self.detector)                                      
        self.matcher.setDescriptorExtractor(self.descriptor)                                
        self.matcher.setDescriptorMatcher(self.create_matcher(self.featureName))        
        self.matcher.setRatio(self.ratioTest) 

    def set_camera_parameters(self, fx, fy, cx, cy):
        self.camera_params = [ fx,  
                        fy, 
                        cx,      
                        cy]    
        self.pnp_detection = PnP_Problem(self.camera_params)

    def load_mesh(self, model_mesh : Model_Mesh):
        self.model_mesh = model_mesh  # load an object mesh    

    def load_points(self, list_model_points):
        if list_model_points is not None:
            self.list_model_points = list_model_points
            self.model_points_loaded = True

    @threaded
    def camera_slot(self, img_left, img_right):
        if self.calculation_in_progress:
            return
        else:
            if self.model_points_loaded and self.model_mesh.loaded:
                self.calculation_in_progress = True
            else:
                return
        
        inliers_most = 0
        frame_final = img_left.copy()

        for model_points in self.list_model_points:
            # Model info
            self.list_points3d_model = model_points._list_points3d_in_ 
            self.descriptors_model = model_points._descriptors_           
            self.keypoints_model = model_points._list_keypoints_

            good_measurement = False

            frame_vis = img_left.copy()

            good_matches = []
            keypoints_scene = []

            good_matches, keypoints_scene = self.matcher.robustMatch(img_left, self.descriptors_model, self.keypoints_model)

            # -- Step 2: Find out the 2D/3D correspondences
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
                                                self.pnpMethod, inliers_idx,
                                                self.iterationsCount, self.reprojectionError, self.confidence )



                if inliers_idx is None:
                    inliers_idx = np.array([])
                
                if len(inliers_idx) < inliers_most:
                    continue
                else:
                    inliers_most = len(inliers_idx)

                for inliers_index in range(len(inliers_idx)):
                    n = inliers_idx[inliers_index][0]         # i-inlier
                    point2d = list_points2d_scene_match[n]     # i-inlier point 2D
                    list_points2d_inliers.append(point2d)           # add i-inlier to list

                draw2DPoints(frame_vis, list_points2d_inliers, self.blue)

                if( len(inliers_idx) >= self.minInliers ):
                    good_measurement = True
            
            l = 5
            pose_points2d = []

            if(good_measurement):
                drawObjectMesh(frame_vis, self.model_mesh, self.pnp_detection, self.green)  # draw current pose

                pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([0,0,0], dtype=np.float64)))  # axis center
                pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([l,0,0], dtype=np.float64)))  # axis x
                pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([0,l,0], dtype=np.float64)))  # axis y
                pose_points2d.append(self.pnp_detection.backproject3DPoint(np.array([0,0,l], dtype=np.float64)))  # axis z
                draw3DCoordinateAxes(frame_vis, pose_points2d)           # draw axes
    #TODO Tady to chce else větev, která vykreslí starou pózu pokud nebyla detekovaná nová, ale musí to být omezené na pár snímků - jen pro zvýšení robustnosti
            frame_final = frame_vis.copy()

        self.detection_update_signal.emit(frame_final)
        self.calculation_in_progress = False


    def create_features(self, featureName, numKeypoints):
        detector = None
        descriptor = None
        if (featureName == "ORB"):
            detector = cv2.ORB_create(numKeypoints)
            descriptor = cv2.ORB_create(numKeypoints)
        elif (featureName == "KAZE"):
            detector = cv2.KAZE_create()
            descriptor = cv2.KAZE_create()
        elif (featureName == "AKAZE"):
            detector = cv2.AKAZE_create()
            descriptor = cv2.AKAZE_create()
        elif (featureName == "BRISK"):
            detector = cv2.BRISK_create()
            descriptor = cv2.BRISK_create()
        elif (featureName == "SIFT"):
            detector = cv2.SIFT_create()
            descriptor = cv2.SIFT_create()

        return detector, descriptor

    def create_matcher(self, featureName):
        if (featureName == "ORB" or featureName == "BRISK" or featureName == "AKAZE" or featureName == "BINBOOST"):
            return cv2.DescriptorMatcher_create("BruteForce-Hamming")
        else:
            return cv2.DescriptorMatcher_create("BruteForce")
