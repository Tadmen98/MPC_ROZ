from abc import ABC, abstractmethod


class PoseDetectorBase(ABC):
    """
    pose
    """

    @abstractmethod
    def load_model_point_cloud(self, file_path):
        pass

    @abstractmethod
    def init_model_point_cloud(self, point_cloud):
        pass

    @abstractmethod
    def create_stereo_corr_obj(self):
        pass

    @abstractmethod
    def gen_scene_point_cloud(self, img_left, img_right):
        pass

    #@Slot()
    @abstractmethod
    def camera_slot(self, img_left, img_right):
        pass

    @abstractmethod
    def find_object_in_3d_scene(self, img_left, img_right):
        pass