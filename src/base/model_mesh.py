from stl import mesh
import numpy as np

class Model_Mesh():
    def __init__(self):
        self.vertices_count = 0
        self.triangles_count = 0
        self.vertices = []
        self.triangles = []
        self.loaded = False
    
    def getVertex(self, position):
        return self.vertices[position]

    def load(self, path):
        self.vertices.clear()
        self.triangles.clear()

        stl_mesh = mesh.Mesh.from_file(path)

        self.vertices = stl_mesh.points.reshape(-1, 3)
        for i in range(0, len(self.vertices), 3):
            self.triangles.append((self.vertices[i], self.vertices[i+1], self.vertices[i+2]))
        self.triangles_count = int(len(self.triangles))
        self.vertices = np.unique(stl_mesh.points.reshape(-1, 3), axis=0)
        
        self.vertices_count = int(len(self.vertices))
        self.loaded = True
