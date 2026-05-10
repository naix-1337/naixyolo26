import sys
import threading
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QCheckBox, QVBoxLayout,
    QHBoxLayout, QLabel, QTextEdit,
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QTextCursor

from main import main as run_main


class DetectionThread(QThread):
    finished = Signal()
    log = Signal(str)

    def __init__(self, debug=True):
        super().__init__()
        self.debug = debug
        self.stop_event = threading.Event()

    def run(self):
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr

        class _Emitter:
            def __init__(self, signal, orig):
                self.signal = signal
                self.orig = orig

            def write(self, text):
                if text:
                    self.signal.emit(text)
                self.orig.write(text)

            def flush(self):
                self.orig.flush()

        sys.stdout = _Emitter(self.log, _orig_stdout)
        sys.stderr = _Emitter(self.log, _orig_stderr)
        try:
            run_main(debug=self.debug, stop_event=self.stop_event)
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        self.finished.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Launcher")
        self.setFixedSize(500, 400)

        layout = QVBoxLayout()

        self.label = QLabel("Click Start to launch YOLO")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.debug_cb = QCheckBox("Debug Mode")
        self.debug_cb.setChecked(True)
        layout.addWidget(self.debug_cb)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedSize(120, 40)
        self.start_btn.clicked.connect(self.on_start)
        btn_layout.addWidget(self.start_btn)

        self.shutdown_btn = QPushButton("Shutdown")
        self.shutdown_btn.setFixedSize(120, 40)
        self.shutdown_btn.clicked.connect(self.on_shutdown)
        btn_layout.addWidget(self.shutdown_btn)

        layout.addLayout(btn_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)
        self.thread = None

    def on_start(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Running...")
        self.label.setText("Detection running — press Q to stop")
        self.showMinimized()

        self.thread = DetectionThread(debug=self.debug_cb.isChecked())
        self.thread.log.connect(self.append_log)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_shutdown(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop_event.set()
            self.thread.wait()
            self.on_finished()

    def append_log(self, text):
        self.log_box.moveCursor(QTextCursor.End)
        self.log_box.insertPlainText(text)

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
