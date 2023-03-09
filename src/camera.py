import cv2
class Camera:

    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self.capture = None

    def connect_camera(self):
        self.capture = cv2.VideoCapture(self.camera_index)

    def camera_read(self):
        if self.capture is not None:
            return self.capture.read()
