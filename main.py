from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.backend import Backend

if __name__ == "__main__":
    app = QApplication()
    backend = Backend()

    main_window = MainWindow(backend)
    main_window.show()

    app.exec()