from PySide6.QtWidgets import QDialogButtonBox, QDialog, QVBoxLayout, QLabel
from PySide6.QtGui import QIcon, QFont

class CustomDialog(QDialog):
    """This class handle dialog pop up with custom window title and custom message"""

    def __init__(
        self, win_title: str, custom_message: str
    ):
        super().__init__()

        self.setWindowTitle(win_title)

        QBtn = (
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)  # type: ignore
        self.buttonBox.rejected.connect(self.reject)  # type: ignore

        self.layout = QVBoxLayout()  # type: ignore
        message = QLabel(custom_message)
        self.layout.addWidget(message)  # type: ignore
        self.layout.addWidget(self.buttonBox)  # type: ignore
        self.setLayout(self.layout)  # type: ignore