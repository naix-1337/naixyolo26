from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import QThread, Signal, Qt

from main import main as run_main


class DetectionThread(QThread):
    finished = Signal()

    def run(self):
        run_main()
        self.finished.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Launcher")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Click Start to launch YOLO")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedSize(120, 40)
        self.start_btn.clicked.connect(self.on_start)
        layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.thread = None

    def on_start(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Running...")
        self.label.setText("Detection running — press Q to stop")
        self.showMinimized()

        self.thread = DetectionThread()
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start")
        self.label.setText("Click Start to launch YOLO")
        self.showNormal()
        self.activateWindow()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
