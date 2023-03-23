from stl import mesh
import numpy as np
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem
from PySide6 import QtWidgets, QtCore
from threading import Thread
from src.camera import Camera
import cv2

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

    def __init__(self):
        super(Backend, self).__init__()
        self.camera_sel_cb_add_items_signal = None
        self.camera = Camera()

    
    def get_model(self):
        """Opens file dialog for user to select stl model.Method
        is called by Load model button.
        """

        #Open file dialog for choosing a file
        name = QtWidgets.QFileDialog.getOpenFileName() 
        self.load_mesh(name)

    @threaded
    def load_mesh(self, name):
        stl_mesh = mesh.Mesh.from_file(name[0])

        points = stl_mesh.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)

        # point_cloud = np.asarray(points) # how to get point cloud for object detection

        mesh_data = MeshData(vertexes=points, faces=faces)
              
        self.update_model_signal.emit(mesh_data)
    
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

    def disconnect_camera(self):
        self.camera.stop()
        