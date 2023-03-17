from stl import mesh
import numpy as np
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem
from PySide6 import QtWidgets, QtCore
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

class Backend(QtCore.QObject):
    update_model_signal = QtCore.Signal(MeshData)

    def __init__(self):
        super(Backend, self).__init__()

    
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

        mesh_data = MeshData(vertexes=points, faces=faces)
              
        self.update_model_signal.emit(mesh_data)
    
        