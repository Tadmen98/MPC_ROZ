import cv2
import numpy as np
from PySide6 import QtCore, QtGui
from src.base.correspondence_matcher import Correspondence_Matcher
from src.base.draw_functions import *
from src.base.model_mesh import Model_Mesh
from src.base.model_points import Model_Points
from src.base.pnp_problem import PnP_Problem


class Model_Registration(QtCore.QObject):
    registration_update_signal = QtCore.Signal(float,float,float)

    def __init__(self, ):
        super(Model_Registration, self).__init__()
        self.end_registration = False
        self.cancel_registration = False

        self.camera_params = [ 385.395,   # fx
                            385.3225,  # fy
                            333.435,      # cx
                            190.276]    # cy

        self.pts = [1, 2, 3, 4, 5, 6, 7, 8] 

        self.model_points = Model_Points()
        self.model_mesh = Model_Mesh()
        self.pnp = PnP_Problem(self.camera_params)

        self.registered = 0
        self.to_be_registered = 8
        self.points2d = []
        self.points3d = []
        
        self.save_path = ""
        self.keypoints_count = 0
        self.extractor_name = "KAZE"

    def is_registrable(self):
        return (self.registered < self.to_be_registered)
    
    def registerPoint(self, point2d, point3d):
        # add correspondence at the end of the vector
        self.points2d.append(point2d)
        self.points3d.append(point3d)
        self.registered += 1

    def reset(self):
        self.registered = 0
        self.to_be_registered = 0
        self.points2d.clear()
        self.points3d.clear()
        self.save_path = ""
        self.keypoints_count = 0
        self.extractor_name = "KAZE"
        self.end_registration = False

    def set_parameters(self, save_path, keypoints_count, extractor_name):
        self.save_path = save_path
        self.keypoints_count = keypoints_count
        self.extractor_name = extractor_name

    def save(self):
        pass

    def start(self, img, model_mesh):
        self.model_mesh = model_mesh
        self.to_be_registered = model_mesh.vertices_count
        self.pts = list(range(1, self.to_be_registered+1))
        matcher = Correspondence_Matcher()
        
        detector, descriptor = self.create_features(self.extractor_name, self.keypoints_count)
        matcher.setFeatureDetector(detector)
        matcher.setDescriptorExtractor(descriptor)

        cv2.namedWindow("Register model", cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback("Register model", self.on_mouse_click, 0)
        
        img_vis = None

        red = (0, 0, 255)
        green = (0,255,0)
        blue = (255,0,0)

        while ( cv2.waitKey(30) < 0 ):
            img_vis = img.copy()

            list_points2d = self.points2d
            list_points3d = self.points3d

            drawPoints(img_vis, list_points2d, list_points3d, red)
            n_regist = self.registered
            if (not self.end_registration and n_regist < len(self.pts)):
                drawCounter(img_vis, self.registered, self.to_be_registered, red)
            else:
                drawText(img_vis, "Registration Complete", green)
                drawCounter(img_vis, self.registered, self.to_be_registered, green)
                break
            
            self.registration_update_signal.emit(self.model_mesh.vertices[self.registered][0], self.model_mesh.vertices[self.registered][1], self.model_mesh.vertices[self.registered][2])
            cv2.imshow("Register model", img_vis)

        list_points2d = self.points2d
        list_points3d = self.points3d

        is_correspondence = self.pnp.estimatePose(list_points3d, list_points2d, cv2.SOLVEPNP_ITERATIVE)
        if ( is_correspondence ):
            list_points2d_mesh = self.pnp.verify_points(model_mesh)
            draw2DPoints(img_vis, list_points2d_mesh, green)

        cv2.imshow("Register model", img_vis)
        cv2.waitKey(0)
        if self.cancel_registration == False:
            keypoints_model = matcher.computeKeyPoints(img)
            descriptors = matcher.computeDescriptors(img, keypoints_model)

            list_points_out = []
            list_points_in = []
            
            for i in range(len(keypoints_model)):
                point2d = np.array(keypoints_model[i].pt)
                point3d = None
                on_surface, point3d = self.pnp.backproject2DPoint(model_mesh, point2d)
                if (on_surface):
                    self.model_points.add_corespondence(point2d, point3d)
                    self.model_points.add_descriptor(descriptors[i])
                    self.model_points.add_keypoint(keypoints_model[i])
                    list_points_in.append(point2d)
                else:
                    self.model_points.add_outlier(point2d)
                    list_points_out.append(point2d)
            
            img_vis = img.copy()


            num = str(len(list_points_in))
            text = "There are " + num + " inliers"
            drawText(img_vis, text, green)

            num = str(len(list_points_out))
            text = "There are " + num + " outliers"
            drawText2(img_vis, text, red)

            drawObjectMesh(img_vis, model_mesh, self.pnp, blue)

            draw2DPoints(img_vis, list_points_in, green)
            draw2DPoints(img_vis, list_points_out, red)

            cv2.imshow("Register model", img_vis)
            cv2.waitKey(0)
        
        cv2.destroyWindow("Register model")
        
        self.registered = 0
        self.end_registration = False
        self.cancel_registration = False
        self.points2d.clear()
        self.points3d.clear()

    def stop(self):
        self.model_points.extractor = self.extractor_name
        self.model_points.save(self.save_path)

    def on_mouse_click(self, event, x, y, int, *void):
        if  (event == cv2.EVENT_LBUTTONUP):
            if (self.is_registrable):
                n_regist = self.registered
                n_vertex = self.pts[n_regist]

                point_2d = (x,y)
                point_3d = self.model_mesh.getVertex(n_vertex-1)

                self.registerPoint(point_2d, point_3d)
                if self.registered == self.to_be_registered:
                    self.end_registration = True
        elif (event == cv2.EVENT_RBUTTONUP):
            self.registered += 1
            if self.end_registration:
                self.cancel_registration = True
            if self.registered == self.to_be_registered:
                    self.end_registration = True



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