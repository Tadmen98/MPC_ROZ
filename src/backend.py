import cv2
from threading import Thread
from src.camera import Camera

def threaded(fn):
    """To use as decorator to make a function call threaded.
    Needs import
    from threading import Thread"""

    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class Backend:
    def __init__(self):
        self.camera_sel_cb_add_items_signal = None
        self.camera = Camera()

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
            # print("still running")
        self.camera.choose_camera(camera_index)
        self.camera.start()

    def disconnect_camera(self):
        self.camera.stop()


