import cv2
import numpy as np

class PoseDetector:
    """
    pose
    """
    def __init__(self):
        self.num_of_disparities = 64
        self.block_size = 15

        self.model_point_cloud = None
        self.stero_bm = None



        self.ppf_matcher = cv2.ppf_match_3d.PPF3DDetector()

    def load_model_point_cloud(self, file_path):
        pass

    def init_model_point_cloud(self, point_cloud):
        self.model_point_cloud = point_cloud

    def create_stereo_corr_obj(self, num_of_disp=64, block_size=15):
        self.stero_bm = cv2.StereoBM_create(numDisparities=num_of_disp, blockSize=block_size)

    def gen_scene_point_cloud(self, img_left, img_right):
        disparity = self.stero_bm.compute(img_left, img_right)

