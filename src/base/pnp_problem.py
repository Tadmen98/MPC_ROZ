import numpy as np
import cv2
from src.base.model_mesh import Model_Mesh
from src.base.ray import Ray
from src.base.triangle import Triangle
from math import sqrt

class PnP_Problem():
    _A_matrix_ = None
    _R_matrix_ = None
    _t_matrix_ = None
    _P_matrix_ = None

    def __init__(self, params : list):
        self._A_matrix_ = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=np.float64).reshape((3,3))   # intrinsic camera parameters
        self._A_matrix_[0, 0] = params[0]       #      [ fx   0  cx ]
        self._A_matrix_[1, 1] = params[1]       #      [  0  fy  cy ]
        self._A_matrix_[0, 2] = params[2]       #      [  0   0   1 ]
        self._A_matrix_[1, 2] = params[3]
        self._A_matrix_[2, 2] = 1
        self._R_matrix_ = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=np.float64).reshape((3,3))   # rotation matrix
        self._t_matrix_ = np.array([0,0,0], dtype=np.float64).reshape((3,1))   # translation matrix
        self._P_matrix_ = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]], dtype=np.float64).reshape((3,4))   # rotation-translation matrix


    def backproject2DPoint(self, model_mesh, point2d):
        point3d = None
        triangles_list = model_mesh.triangles

        lambd = 1
        u = point2d[0]
        v = point2d[1]

        point2d_vec = np.array([[u * lambd],[v * lambd],[lambd]], dtype=np.float64)

        X_w = np.linalg.pinv((self._A_matrix_ @ self._P_matrix_)) @ point2d_vec
        X_w = np.array([X_w[0]/X_w[3],X_w[1]/X_w[3],X_w[2]/X_w[3]], dtype=np.float64)
        
        C_op = (np.linalg.inv(self._R_matrix_) * (-1)) @ self._t_matrix_ 

        ray = X_w - C_op
        ray = ray / cv2.norm(ray)

        R = Ray(C_op, ray)

        intersections_list = []

        for i in range(len(triangles_list)):
            V0 = triangles_list[i][0]
            V1 = triangles_list[i][1]
            V2 = triangles_list[i][2]

            T = Triangle(V0, V1, V2)

            out = None
            ret, out = self.check_intersection(R, T)
            if(ret):
                tmp_pt = R.p0 + out*R.p1
                intersections_list.append(tmp_pt)

        if (len(intersections_list) != 0):
            point3d = get_nearest_3D_point(intersections_list, R.p0)
            return True, point3d
        else:
            return False, point3d

    def check_intersection(self, ray, triangle):
        """Based on Opencv example"""
        EPSILON = 0.000001
        out = None
        V1 = triangle.v0
        V2 = triangle.v1
        V3 = triangle.v2

        O = ray.p0
        D = ray.p1

        #Find vectors for two edges sharing V1
        e1 = np.subtract(V2, V1)
        e2 = np.subtract(V3, V1)

        # Begin calculation determinant - also used to calculate U parameter
        P = np.cross(D.reshape((3,1)), e2.reshape((3,1)), axis=0)

        # If determinant is near zero, ray lie in plane of triangle
        det = np.sum(e1.reshape((3,1)) * P.reshape((3,1)))

        #NOT CULLING
        if(det > -EPSILON and det < EPSILON):
            return False, out
        inv_det = 1.0 / det

        #calculate distance from V1 to ray origin
        T = np.subtract(O.reshape((3,1)), V1.reshape((3,1)))

        #Calculate u parameter and test bound
        u = np.sum(T.reshape((3,1)) * P.reshape((3,1))) * inv_det

        #The intersection lies outside of the triangle
        if(u < 0.0 or u > 1.0):
            return False, out

        #Prepare to test v parameter
        Q = np.cross(T.reshape((3,1)), e1.reshape((3,1)), axis=0)

        #Calculate V parameter and test bound
        v = np.sum(D.reshape((3,1)) * Q.reshape((3,1))) * inv_det

        #The intersection lies outside of the triangle
        if(v < 0.0 or (u + v)  > 1.0):
            return False, out

        t = np.sum(e2.reshape((3,1)) * Q.reshape((3,1))) * inv_det

        if(t > EPSILON):
            #ray intersection
            out = t
            return True, out
        
        # No hit, no win
        return False, out

    def verify_points(self, model_mesh):
        verified_points_2d = []
        for i in range(model_mesh.vertices_count):
            point3d = model_mesh.getVertex(i)
            point2d = self.backproject3DPoint(point3d)
            verified_points_2d.append(point2d)
        

        return verified_points_2d

    def backproject3DPoint(self, point3d):
        point3d_vec = np.array([point3d[0],point3d[1],point3d[2],1], dtype=np.float64).reshape((4,1))
        point2d_vec = self._A_matrix_ @ self._P_matrix_ @ point3d_vec

        point2d = [0,0]
        if point2d_vec[2] != 0:
            point2d[0] = int(point2d_vec[0] / point2d_vec[2])
            point2d[1] = int(point2d_vec[1] / point2d_vec[2])
        else:
            point2d[0] = 0
            point2d[1] = 0

        return tuple(point2d)

    def estimatePose(self, list_points3d, list_points2d, flags):
        distCoeffs = np.zeros((4, 1), dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        useExtrinsicGuess = False

        list_points2d = np.array(list_points2d, dtype=np.float64)
        list_points3d = np.array(list_points3d, dtype = np.float64)
        
        ret, rvec, tvec = cv2.solvePnP( list_points3d, list_points2d, self._A_matrix_, distCoeffs, rvec, tvec,
                                            useExtrinsicGuess, flags)

        cv2.Rodrigues(rvec, self._R_matrix_)
        self._t_matrix_ = tvec

        self.set_P_matrix(self._R_matrix_, self._t_matrix_)

        return ret

    def estimatePoseRANSAC(self, list_points3d, list_points2d, flags, inliers, iterationsCount, reprojectionError, confidence):
        distCoeffs = np.array([-0.0383038,0.0234572,0,0,-0.00448757,-0.00029159,0,0], dtype=np.float64)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        list_points2d = np.array(list_points2d)
        list_points3d = np.array(list_points3d)
        useExtrinsicGuess = False
        
        ret, rvec, tvec, inliers = cv2.solvePnPRansac( list_points3d, list_points2d, self._A_matrix_, distCoeffs, rvec, tvec,
                            useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                            inliers, flags )

        cv2.Rodrigues(rvec, self._R_matrix_)
        self._t_matrix_ = tvec

        self.set_P_matrix(self._R_matrix_, self._t_matrix_)
        return inliers


    def set_P_matrix(self, R_matrix, t_matrix):
        self._P_matrix_[0,0] = R_matrix[0,0]
        self._P_matrix_[0,1] = R_matrix[0,1]
        self._P_matrix_[0,2] = R_matrix[0,2]
        self._P_matrix_[1,0] = R_matrix[1,0]
        self._P_matrix_[1,1] = R_matrix[1,1]
        self._P_matrix_[1,2] = R_matrix[1,2]
        self._P_matrix_[2,0] = R_matrix[2,0]
        self._P_matrix_[2,1] = R_matrix[2,1]
        self._P_matrix_[2,2] = R_matrix[2,2]
        self._P_matrix_[0,3] = t_matrix[0]
        self._P_matrix_[1,3] = t_matrix[1]
        self._P_matrix_[2,3] = t_matrix[2]


def get_nearest_3D_point(points_list, origin):
    if len(points_list) == 1:
        return points_list[0]
    
    p1 = points_list[0]
    p2 = points_list[1]

    d1 = sqrt( pow(p1[0]-origin[0], 2) + pow(p1[1]-origin[1], 2) + pow(p1[2]-origin[2], 2) )
    d2 = sqrt( pow(p2[0]-origin[0], 2) + pow(p2[1]-origin[1], 2) + pow(p2[2]-origin[2], 2) )

    if(d1 < d2):
        return p1
    else:
        return p2