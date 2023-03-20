from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt
import numpy as np
import cv2


class Camera(QThread):
    image_update = Signal(np.ndarray, np.ndarray)

    def __init__(self):
        self.camera_index = None
        self.capture = None
        super().__init__()

    def choose_camera(self, camera_index : int):
        self.camera_index = camera_index

    def run(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        if self.capture is None:
            self.ThreadActive = False
            return False
        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = self.capture.read()
            if ret:
                frame_left, frame_right = np.split(frame, 2, axis=1)
                self.image_update.emit(frame_left, frame_right)

    def stop(self):
        self.ThreadActive = False
        self.camera_index = None
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        self.capture = None
        self.quit()

