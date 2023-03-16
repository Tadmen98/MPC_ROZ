from PySide6.QtCore import QThread, Signal
import cv2


class Camera(QThread):
    ImageUpdate = Signal(int, int)

    def __init__(self):
        self.camera_index = None
        self.capture = None
        super().__init__()

    def connect_camera(self, camera_index : int):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(self.camera_index)

    def run(self):
        if self.capture is None:
            self.ThreadActive = False
            return False
        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = self.capture.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR)
                print(type(img))

    def stop(self):
        self.ThreadActive = False
        self.camera_index = None
        self.capture.release()
        cv2.destroyAllWindows()
        self.capture = None
        self.quit()

