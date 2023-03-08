import cv2


class Camera:
    pass


def find_aviable_cameras():
    camera_indexes = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            camera_indexes.append(i)
        else:
            return camera_indexes