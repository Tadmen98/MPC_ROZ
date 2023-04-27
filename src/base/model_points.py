import numpy as np
import cv2
import pickle

class Model_Points():
    def __init__(self):
        self._n_correspondences_ = 0
        self._list_keypoints_ = []
        self.points2d_in = [] 
        self.points2d_out = []
        self.points3d_in = []
        self._descriptors_ = [] 
        self._training_img_path_ = None
        self.loaded = False


    def add_corespondence(self, point2d, point3d):
        self.points2d_in.append(point2d)
        self.points3d_in.append(point3d)
        self._n_correspondences_ += 1

    def add_outlier(self, point2d):
        self.points2d_out.append(point2d)


    def add_descriptor(self, descriptor):
        """Descriptor is Matrix"""
        self._descriptors_.append(descriptor)

    def add_keypoint(self, kp : cv2.KeyPoint):
        self._list_keypoints_.append(kp)

    def save(self, path):
        points3d_array = np.array(self.points3d_in, dtype=np.float32)
        points2d_array = np.array(self.points2d_in, dtype=np.float32)

        file = open(path, 'w+')
        file.close()
        
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

        storage.write("points_3d", points3d_array)
        storage.write("points_2d", points2d_array)
        
        index = []
        for point in self._list_keypoints_:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, 
                point.class_id) 
            index.append(temp)
        with open(path + "keypoints.pickle", "wb") as f:
            pickle.dump(index, f)

        desc = np.array(self._descriptors_, dtype=np.float32)
        storage.write("descriptors", desc)
        storage.write("training_image_path", self._training_img_path_)

        storage.release()

    def load(self, path : str):
        points3d_array = np.empty((0, 3), dtype=np.float32)

        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        points3d_mat = storage.getNode("points_3d").mat()
        points3d_array = np.asarray(points3d_mat)
        self._list_points3d_in_ = points3d_array.tolist()

        self._descriptors_ = storage.getNode("descriptors").mat()

        index = None
        with open(path + "keypoints.pickle", "rb") as f:
            index = pickle.load(f)

        for point in index:
            temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2], 
                                    response=point[3], octave=point[4], class_id=point[5]) 
            self._list_keypoints_.append(temp)
        
        training_image_path_node = storage.getNode("training_image_path")
        if not training_image_path_node.empty():
            self._training_img_path_ = training_image_path_node.string()

        storage.release()
        self.loaded = True