from PySide6.QtWidgets import QMainWindow

class MainWindow(QMainWindow):

    def __init__(self, backend):
        super().__init__()

        self.backend = backend

        self.setup_ui()

    def setup_ui(self):
        pass

