import cv2
import numpy as np
from src.base.pose_detector_base import PoseDetectorBase
from PySide6 import QtCore
from pyqtgraph.opengl import  MeshData,GLScatterPlotItem


class PoseDetector(PoseDetectorBase):
    """
    pose
    """
    update_scene_signal = QtCore.Signal(GLScatterPlotItem)

    def __init__(self):
        super(PoseDetector, self).__init__()
        self.num_of_disparities = 64
        self.block_size = 15

        self.focal_length = 385.4825  # in pixels for HD720, for HD1080 use 1400
        self.baseline = 62.8594  # m

        self.ppc_x = 0 #  x-coordinates of the principal point
        self.ppc_y = 0 #  y-coordinates of the principal point

        self.ppc_x_left = 0 #   x-coordinates of the principal points in the left image
        self.ppc_x_right = 0 #   x-coordinates of the principal points in the right image

        self.model_point_cloud = None
        self.stero_bm = None

        self.dc1 = None
        self.dc2 = None
        self.cm1 = None
        self.cm2 = None
        self.calculation_in_progress = False
        self.create_stereo_corr_obj()


        #self.ppf_matcher = cv2.ppf_match_3d.PPF3DDetector()

    def load_model_point_cloud(self, file_path):
        pass

    def init_model_point_cloud(self, point_cloud):
        self.model_point_cloud = point_cloud

    def set_camera_parameters(self, focal_length, baseline, ppc_x, ppc_y, ppc_x_left, ppc_x_right):
        self.focal_length = focal_length
        self.baseline = baseline
        self.ppc_x = ppc_x
        self.ppc_y = ppc_y
        self.ppc_x_left = ppc_x_left
        self.ppc_x_right = ppc_x_right

    def create_stereo_corr_obj(self, num_of_disp=64, block_size=15):
        self.stero_bm = cv2.StereoBM_create(numDisparities=num_of_disp, blockSize=block_size)

    def gen_scene_point_cloud(self, img_left, img_right):
        disparity = self.stero_bm.compute(img_left, img_right)
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(disparity)
        focal_length = 0.8 * img_left.shape[1]  # assume a horizontal field of view of 60 degrees
        baseline = 0.1  # assume a baseline of 10 cm
        Q = np.float32([[1, 0, 0, 1.34320702e+02],
                        [0, -1, 0, 1.20138880e+02],
                        [0, 0, 0, 1],
                        [0, 0, 8.19142722e-02, 0]])
        #Q = np.float32([[1, 0, 0, -0.5 * img_left.shape[1]],
        #                [0, -1, 0, 0.5 * img_left.shape[0]],
        #                [0, 0, 0, -focal_length],
        #                [0, 0, 1 / baseline, 0]])
        points = cv2.reprojectImageTo3D(disparity, Q)
        points = points.reshape(-1, 3)
        #for point in points:
            #print(point)
        #points = np.load('points.npy')

        # Extract x, y, z coordinates from points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Create position array for PyQtGraph
        pos = np.empty((len(points), 3))
        pos[:, 0] = x
        pos[:, 1] = y
        pos[:, 2] = z

        # Create color array for PyQtGraph
        color = np.empty((len(points), 4))
        color[:, 0] = 1.0  # red color
        color[:, 1] = 0.0
        color[:, 2] = 0.0
        color[:, 3] = 1.0  # full opacity
        #pos = np.random.normal(size=(n, 3))
        #size = np.ones((points_3d)) * 0.1
        #color = (1.0, 0.0, 0.0, 1.0)  # red color
        #scatter = GLScatterPlotItem(pos=points_3d, pxMode=False)
        scatter = GLScatterPlotItem(pos=pos, color=color, size=0.1, pxMode=False)
        self.update_scene_signal.emit(scatter)
        #cv2.imshow("Disparity", points_3d)
        #mask = disparitydisparity > disparity.min()
        #points_3d = points_3d[mask]

        return points

    #@Slot()
    def camera_slot(self, img_left, img_right):
        if self.calculation_in_progress:
            return
        else:
            self.calculation_in_progress = True
        
        img_size = (img_left.shape[1],img_left.shape[0])
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Define camera coefs, should move to init
        left_camera_matrix = np.array([[385.395, 0, 333.435], [0, 385.3225, 190.276], [0, 0, 1]], dtype=np.float64)
        right_camera_matrix = np.array([[385.4825, 0, 337.2475], [0, 385.445, 186.01475], [0, 0, 1]], dtype=np.float64)
        distortion_coefficients_left = np.array([-0.0383038,0.0234572,0,0,-0.00448757,-0.00029159,0,0], dtype=np.float64)
        distortion_coefficients_right = np.array([-0.0364626,0.0235784,0,0,-0.0101603,0.0044135,0,0], dtype=np.float64)

        # Define the relative position and orientation of the two cameras
        R = np.array([[0.00245413, 0, 0], [0, 0.00353464, 0], [0, 0, 0.0012617]], dtype=np.float64)
        T = np.array([-self.baseline, 0, 0], dtype=np.float64)

        # Compute the rectification maps and disparity-to-depth mapping matrix
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, distortion_coefficients_left, right_camera_matrix, distortion_coefficients_right, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
        
        map1, map2 = cv2.initUndistortRectifyMap(left_camera_matrix, distortion_coefficients_left, R1, P1, img_size, cv2.CV_32FC1)
        rectified_left = cv2.remap(gray_left, map1, map2,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)

        map1, map2 = cv2.initUndistortRectifyMap(right_camera_matrix, distortion_coefficients_right, R2, P2, img_size, cv2.CV_32FC1)
        rectified_right = cv2.remap(gray_right, map1, map2, cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=11)

        # compute the disparity map
        disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
        #disparity = np.clip(disparity, 0, 255)
        #disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #cv2.imshow("disp",disparity_normalized)

        # Compute the 3D point cloud
        depth = cv2.reprojectImageTo3D(disparity, Q)
        mask = disparity > disparity.min()
        points = depth[mask]

        points = points.reshape(-1, 3)
        colors = img_left.reshape(-1, 3)

        # Extract x, y, z coordinates from points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x = [i for index,i in enumerate(x) if points[index, 2] < 500]
        y = [i for index,i in enumerate(y) if points[index, 2] < 500]
        z = [i for index,i in enumerate(z) if points[index, 2] < 500]

        # Create position array for PyQtGraph
        pos = np.empty((len(z), 3))
        pos[:, 0] = x
        pos[:, 1] = y
        pos[:, 2] = z

        # Create color array for PyQtGraph
        color = np.empty((len(points), 4))
        color[:, 0] = 1.0  # red color
        color[:, 1] = 0.0
        color[:, 2] = 0.0
        color[:, 3] = 1.0  # full opacity

        scatter = GLScatterPlotItem(pos=pos, color=color, size=0.1, pxMode=False)
        self.update_scene_signal.emit(scatter)
        #self.calculation_in_progress = False

    def find_object_in_3d_scene(self, img_left, img_right):
        scene_point_cloud = self.gen_scene_point_cloud(img_left, img_right)
        
        # Some logic

        object_pose = None

        return object_pose

    def reset_calculation_in_progress(self):
        print("RESETING")
        self.calculation_in_progress = False


