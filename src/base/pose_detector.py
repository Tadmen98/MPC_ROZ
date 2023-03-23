import cv2
import numpy as np
from pose_detector_base import PoseDetectorBase


class PoseDetector(PoseDetectorBase):
    """
    pose
    """
    def __init__(self):
        self.num_of_disparities = 64
        self.block_size = 15

        self.focal_length = 700  # in pixels for HD720, for HD1080 use 1400
        self.baseline = 0.063  # m

        self.ppc_x = 0 #  x-coordinates of the principal point
        self.ppc_y = 0 #  y-coordinates of the principal point

        self.ppc_x_left = 0 #   x-coordinates of the principal points in the left image
        self.ppc_x_right = 0 #   x-coordinates of the principal points in the right image

        self.model_point_cloud = None
        self.stero_bm = None



        self.ppf_matcher = cv2.ppf_match_3d.PPF3DDetector()

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

        focal_length = 0.8 * img_left.shape[1]  # assume a horizontal field of view of 60 degrees
        baseline = 0.1  # assume a baseline of 10 cm
        Q = np.float32([[1, 0, 0, -0.5 * img_left.shape[1]],
                        [0, -1, 0, 0.5 * img_left.shape[0]],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1 / baseline, 0]])
        points_3d = cv2.reprojectImageTo3D(disparity, Q)

        mask = disparity > disparity.min()
        points_3d = points_3d[mask]

        return points_3d

    #@Slot()
    def camera_slot(self, img_left, img_right):
        object_coordinates = self.find_object_in_3d_scene(img_left, img_right)
        #self.update_object_pose_signal.emit()

    def find_object_in_3d_scene(self, img_left, img_right):
        scene_point_cloud = self.gen_scene_point_cloud(img_left, img_right)

        # Some logic

        object_pose = None

        return object_pose



