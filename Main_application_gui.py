import sys
import os
import json
import time
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QSplitter, QMessageBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from pypylon import pylon
from InvariantTM import template_crop, invariant_match_template
from live_tester_optimized import analyze_frame_cv, match


class CameraWorker(QThread):
    new_frame = Signal(np.ndarray)
    matched_frame = Signal(np.ndarray)
    measured_image = Signal(np.ndarray, list)

    def __init__(self, template_rgb, template_data, auto_mode=True):
        super().__init__()
        self.running = False
        self.auto_mode = auto_mode
        self.template_rgb = template_rgb
        self.template_data = template_data
        self.camera = None

    def run(self):
        self.running = True
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        if self.auto_mode:
            self.camera.TriggerSelector.SetValue('FrameStart')
            self.camera.TriggerMode.SetValue('On')
            self.camera.TriggerSource.SetValue('Line1')
        else:
            self.camera.TriggerMode.SetValue('Off')

        self.camera.ExposureAuto.SetValue('Off')
        self.camera.ExposureTime.SetValue(300)
        self.camera.GainAuto.SetValue('Off')
        self.camera.Gain.SetValue(30.0)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while self.running and self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result.GrabSucceeded():
                frame = converter.Convert(grab_result).GetArray()
                self.new_frame.emit(frame)

                if self.auto_mode:
                    matched, flg = match(frame, self.template_rgb)
                    if flg:
                        self.matched_frame.emit(matched)
                        output = analyze_frame_cv(matched, self.template_data, scale=4)
                        results = self.extract_results(output)
                        self.measured_image.emit(output, results)
            grab_result.Release()

        self.camera.StopGrabbing()
        self.camera.Close()

    def stop(self):
        self.running = False
        self.wait()

    def extract_results(self, output):
        return [("H1", 1.23, 1.22, 0.01, True), ("V1", 0.78, 0.79, -0.01, True)]  # dummy


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Matcher & Measurement")
        self.template_rgb = None
        self.template_data = None
        self.template_path = None
        self.worker = None

        self.init_ui()
        self.load_last_template()

    def init_ui(self):
        # Live and output preview
        self.live_label = QLabel("Live Preview")
        self.output_label = QLabel("Output Preview")

        # Buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.capture_btn = QPushButton("Manual Capture")
        self.load_template_btn = QPushButton("Load Template")
        self.edit_template_btn = QPushButton("Edit Template")

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.manual_capture)
        self.load_template_btn.clicked.connect(self.select_template)
        self.edit_template_btn.clicked.connect(self.edit_template)

        # Tables
        self.measure_table = QTableWidget(0, 5)
        self.measure_table.setHorizontalHeaderLabels(["Name", "Design", "Current", "Δ", "Status"])
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["Time", "Name", "Design", "Current", "Δ", "Status"])

        # Layouts
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.live_label)
        left_layout.addWidget(self.output_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.start_btn)
        right_layout.addWidget(self.stop_btn)
        right_layout.addWidget(self.capture_btn)
        right_layout.addWidget(self.load_template_btn)
        right_layout.addWidget(self.edit_template_btn)
        right_layout.addWidget(QLabel("Measurements"))
        right_layout.addWidget(self.measure_table)
        right_layout.addWidget(QLabel("History"))
        right_layout.addWidget(self.history_table)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def display_image(self, label, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_camera(self):
        if self.template_rgb is None or self.template_data is None:
            QMessageBox.warning(self, "Error", "Load a template first.")
            return
        self.worker = CameraWorker(self.template_rgb, self.template_data)
        self.worker.new_frame.connect(lambda img: self.display_image(self.live_label, img))
        self.worker.matched_frame.connect(lambda img: self.display_image(self.output_label, img))
        self.worker.measured_image.connect(self.update_measurement_results)
        self.worker.start()

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

    def manual_capture(self):
        QMessageBox.information(self, "Manual Mode", "Manual capture not yet implemented.")

    def select_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Template Image", "./templates", "Image Files (*.bmp *.png *.jpg)")
        if path:
            base = os.path.splitext(path)[0]
            json_path = base + ".json"
            if not os.path.exists(json_path):
                QMessageBox.warning(self, "Error", f"Missing JSON file: {json_path}")
                return
            self.template_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            with open(json_path, "r") as f:
                self.template_data = json.load(f)
            self.template_path = base
            with open("last_template.txt", "w") as f:
                f.write(base)

    def edit_template(self):
        if self.template_path:
            os.system(f"python template\ maker.py \"{self.template_path}.bmp\"")
        else:
            QMessageBox.warning(self, "No Template", "Load a template first.")

    def load_last_template(self):
        if os.path.exists("last_template.txt"):
            with open("last_template.txt", "r") as f:
                self.template_path = f.read().strip()
            if os.path.exists(self.template_path + ".bmp") and os.path.exists(self.template_path + ".json"):
                self.template_rgb = cv2.cvtColor(cv2.imread(self.template_path + ".bmp"), cv2.COLOR_BGR2RGB)
                with open(self.template_path + ".json", "r") as f:
                    self.template_data = json.load(f)
            else:
                self.select_template()
        else:
            self.select_template()

    def update_measurement_results(self, image, results):
        self.display_image(self.output_label, image)
        self.measure_table.setRowCount(0)
        for i, (name, design, current, delta, status) in enumerate(results):
            self.measure_table.insertRow(i)
            for j, val in enumerate([name, design, current, delta, "OK" if status else "FAIL"]):
                self.measure_table.setItem(i, j, QTableWidgetItem(str(val)))

            # Append to history
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            for j, val in enumerate([datetime.now().strftime("%H:%M:%S"), name, design, current, delta, "OK" if status else "FAIL"]):
                self.history_table.setItem(row, j, QTableWidgetItem(str(val)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec())
