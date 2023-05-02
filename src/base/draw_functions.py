import cv2
from math import atan2, sin, cos

fontFace = cv2.FONT_ITALIC
fontScale = 0.75
thickness_font = 2

lineType = 8
radius = 4

def drawText(image, text, color):
    cv2.putText(image, text, (25,100), fontFace, fontScale, color, thickness_font, 8)

def drawText2(image, text, color):
    cv2.putText(image, text, (25,75), fontFace, fontScale, color, thickness_font, 8)

def drawCounter(image, n, n_max, color):
    n_str = str(n)
    n_max_str = str(n_max)
    text = n_str + " of " + n_max_str + " points"
    cv2.putText(image, text, (50,50), fontFace, fontScale, color, thickness_font, 8)

def drawPoints( image, list_points_2d, list_points_3d, color):
    for i in range(len(list_points_2d)):
        point_2d = list(list_points_2d[i])
        point_3d = list(list_points_3d[i])

        cv2.circle(image, point_2d, radius, color, -1, lineType )

        idx = str(i+1)
        x = str(point_3d[0])
        y = str(point_3d[1])
        z = str(point_3d[2])

        point_2d[0] = point_2d[0] + 10
        point_2d[1] = point_2d[1] - 10

def draw2DPoints(image, list_points, color):
    for i in range(len(list_points)):
        x = list_points[i][0]
        y = list_points[i][1]
        point_2d = (int(x),int(y))

        cv2.circle(image, point_2d, radius, color, -1, lineType )

def drawObjectMesh(image, model_mesh, pnp, color):
    list_triangles = model_mesh.triangles
    for i in range(len(list_triangles)):
        tmp_triangle = list_triangles[i]

        point_3d_0 = tmp_triangle[0]
        point_3d_1 = tmp_triangle[1]
        point_3d_2 = tmp_triangle[2]

        point_2d_0 = pnp.backproject3DPoint(point_3d_0)
        point_2d_1 = pnp.backproject3DPoint(point_3d_1)
        point_2d_2 = pnp.backproject3DPoint(point_3d_2)

        cv2.line(image, point_2d_0, point_2d_1, color, 1)
        cv2.line(image, point_2d_1, point_2d_2, color, 1)
        cv2.line(image, point_2d_2, point_2d_0, color, 1)


