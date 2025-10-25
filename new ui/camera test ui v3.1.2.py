# merged_camera_and_servo_ui.py
# Combined: original camera test UI (unchanged) + ServoControlPanel on the right
# Requirements: PySide6, qdarkstyle, pypylon, ultralytics, pymodbus, opencv-python, numpy
# Run: python merged_camera_and_servo_ui.py

import sys
import cv2
import time
import threading
from typing import Callable, Optional
import numpy as np
from pypylon import pylon
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QGridLayout, QMainWindow, QFrame, QSizePolicy,
    QLineEdit, QGroupBox, QFormLayout, QSpacerItem, QSizePolicy as QSP
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
import qdarkstyle

from ultralytics import YOLO
from pymodbus.client import ModbusTcpClient
import os
import json

# -----------------------------
# Config (keep UI same; tweak internals)
# -----------------------------
CAMERA_VIEWPORT_SIZE = QSize(480, 320)
WIDTH  = 1920
HEIGHT = 1200
TC_WIDTH  = 5328
TC_HEIGHT = 4608
model_path = "D:/program/brush_knowledge/weights/best.pt"
#model = YOLO(model_path)
#class_names = model.names
# UI paint throttle (frames-per-second)
TARGET_UI_FPS = 30   # set to 24 or 30 as you like

# -----------------------------
# Camera UI code (unchanged in behaviour / layout)
# -----------------------------

camera_index_list = [0,1,2,3,4,5]

with open("configuration.json", "r") as file:
    json_data = json.load(file)

camera_index_list = json_data["camera_index_list"]

FILE_PATH = "camera_data.txt"

def normal_cam_offset_value_write(cam_index, value):
    cam_offset_value_write("Normal_cam:"+str(cam_index), value)

def tele_cam_offset_value_write(cam_index, value):
    cam_offset_value_write("Tele_cam:"+str(cam_index), value)
    
def cam_offset_value_write(cam_index, value):
    """Write or update a camera index and its value in the file."""
    data = {}

    # Read existing data if file exists
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f:
            for line in f:
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    data[parts[0]] = parts[1]

    # Update or add new entry
    data[str(cam_index)] = str(value)

    # Write back to file
    with open(FILE_PATH, "w") as f:
        for key, val in data.items():
            f.write(f"{key}={val}\n")

def normal_cam_offset_value_read(cam_index):
    cam_offset_value_read("Normal_cam:"+str(cam_index))

def tele_cam_offset_value_read(cam_index):
    cam_offset_value_read("Tele_cam:"+str(cam_index))
    
def cam_offset_value_read(cam_index):
    """Read a camera index value from the file."""
    if not os.path.exists(FILE_PATH):
        return None

    with open(FILE_PATH, "r") as f:
        for line in f:
            parts = line.strip().split("=", 1)
            if len(parts) == 2 and parts[0] == str(cam_index):
                return parts[1]
    return None

class FramePacket:
    __slots__ = ("frame", "ts")
    def __init__(self, frame):
        self.frame = frame
        self.ts = time.time()

class UVC_CameraThread(QThread):
    frame_ready = Signal(object)   # emits FramePacket
    fps_updated = Signal(float)
    error_signal = Signal(str)

    def __init__(self, index: int, width=WIDTH, height=HEIGHT):
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self._running = False
        self._cap = None

    def run(self):
        self._cap = cv2.VideoCapture(camera_index_list[self.index], cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self.error_signal.emit(f"UVC {self.index}: failed to open")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self._cap.set(cv2.CAP_PROP_GAIN, 5)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        last = time.time()
        frames = 0
        first_frame=True
        update_threshold = 10_00000
        Test_frame=None
        background=None
        obj_present=False
        
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.error_signal.emit(f"UVC {self.index}: read failed")
                break
            '''
            if first_frame:
                Test_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                background = cv2.cvtColor(Test_frame, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(background, (21, 21), 0)
                first_frame=False

            obj_present=False
            Test_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            gray = cv2.cvtColor(Test_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)

            diff = cv2.absdiff(background, blurred)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            motion_level = np.sum(thresh)

            if motion_level < update_threshold:
                background = cv2.addWeighted(background, 0.9, blurred, 0.1, 0)
                cv2.putText(frame, "Updating background...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                obj_present=False
            else:
                background = cv2.addWeighted(background, 0.9, blurred, 0.1, 0)
                cv2.putText(frame, "Motion Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                obj_present=True'''

            # if obj_present:
            #     try:
            #         results = model(frame)
            #         detections = results[0]
            #         for box in detections.boxes:
            #             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            #             conf = box.conf[0].item()
            #             class_id = int(box.cls[0].item())
            #             class_name = class_names[class_id]
            #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            #             label = f"{class_name}: {conf:.2f}"
            #             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            #     except Exception as e:
            #         # inference errors shouldn't crash camera thread
            #         pass

            self.frame_ready.emit(FramePacket(frame))
            frames += 1
            now = time.time()
            if now - last >= 1.0:
                self.fps_updated.emit(frames / (now - last))
                frames = 0
                last = now

        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def stop(self):
        self._running = False
        self.wait(1500)

class BAS_CameraThread(QThread):
    frame_ready = Signal(object)   # emits FramePacket
    fps_updated = Signal(float)
    error_signal = Signal(str)

    def __init__(self, index: int, width=TC_WIDTH, height=TC_HEIGHT):
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self._running = False
        self._camera: Optional[pylon.InstantCamera] = None

    def run(self):
        try:
            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            if len(devices) <= self.index:
                self.error_signal.emit(f"Basler {self.index}: device not found")
                return

            self._camera = pylon.InstantCamera(tlf.CreateDevice(devices[self.index]))
            self._camera.Open()
            try:
                self._camera.Width.SetValue(self.width)
                self._camera.Height.SetValue(self.height)
            except Exception as e:
                self.error_signal.emit(f"Basler {self.index}: resolution set warning: {e}")

            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._running = True

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat   = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment  = pylon.OutputBitAlignment_MsbAligned

            last = time.time()
            frames = 0

            while self._running and self._camera.IsGrabbing():
                grab = self._camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab.GrabSucceeded():
                    img = converter.Convert(grab)
                    frame = img.GetArray()
                    self.frame_ready.emit(FramePacket(frame))
                    frames += 1
                    now = time.time()
                    if now - last >= 1.0:
                        self.fps_updated.emit(frames / (now - last))
                        frames = 0
                        last = now
                else:
                    self.error_signal.emit(f"Basler {self.index}: grab failed")
                grab.Release()
        except Exception as e:
            self.error_signal.emit(f"Basler {self.index}: {e}")
        finally:
            self._shutdown()

    def _shutdown(self):
        self._running = False
        try:
            if self._camera:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                if self._camera.IsOpen():
                    self._camera.Close()
        except Exception:
            pass
        self._camera = None

    def stop(self):
        self._running = False
        self.wait(1500)

class _BaseCameraWidget1(QWidget):
    """
    Keeps the same look-and-feel (label, image, fps, +/- zoom, Start/Stop),
    but rendering is throttled via a QTimer at TARGET_UI_FPS. Acquisition
    can run at full rate on its own thread.

    You can inject a frame processor with set_processor(callable).
    """
    def __init__(self, label_text: str):
        """ def __init__: Describe purpose. """
        super().__init__()
        self.label_text = label_text
        self.zoom_scale = 0.05
        self._latest_packet: Optional[FramePacket] = None
        self._processor: Optional[Callable[[FramePacket], FramePacket]] = None
        self._timer: Optional[QTimer] = None
        self._thread: Optional[QThread] = None
        self._build_ui()
        self._setup_paint_timer()

    def _build_ui(self):
        """ def _build_ui: Defines a UI widget/window. """
        layout = QVBoxLayout()

        self.label = QLabel(self.label_text)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.image_label = QLabel("No Frame")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(160, 120)
        self.image_label.setMaximumSize(CAMERA_VIEWPORT_SIZE)
        self.image_label.setFrameShape(QFrame.Box)
        layout.addWidget(self.image_label)

        self.fps_label = QLabel("FPS: 0.0")
        layout.addWidget(self.fps_label)

        zoom_layout = QHBoxLayout()
        self.zoom_in_button  = QPushButton("+")
        self.zoom_out_button = QPushButton("-")
        self.zoom_in_button.clicked.connect(self._zoom_in)
        self.zoom_out_button.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_in_button)
        layout.addLayout(zoom_layout)

        self.stream_btn = QPushButton("Start Stream")
        self.stream_btn.setCheckable(True)
        self.stream_btn.toggled.connect(self._toggle_stream)
        layout.addWidget(self.stream_btn)
        self.stream_btn.setText("Stop Stream")
        self._start_thread()

        self.setLayout(layout)

    def _setup_paint_timer(self):
        """ def _setup_paint_timer: Describe purpose. """
        self._timer = QTimer(self)
        interval_ms = int(1000 / max(1, int(TARGET_UI_FPS)))
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._paint_latest)
        self._timer.start()

    def _zoom_in(self):
        """ def _zoom_in: Describe purpose. """
        self.zoom_scale = min(3.0, self.zoom_scale + 0.01)

    def _zoom_out(self):
        """ def _zoom_out: Describe purpose. """
        self.zoom_scale = max(0.05, self.zoom_scale - 0.01)

    def _toggle_stream(self, checked: bool):
        """ def _toggle_stream: Describe purpose. """
        if checked:
            self.stream_btn.setText("Stop Stream")
            self._start_thread()
        else:
            self.stream_btn.setText("Start Stream")
            self._stop_thread()

    # ---- hooks to be implemented by subclasses ----
    def _make_thread(self) -> QThread:
        """ def _make_thread: Describe purpose. """
        raise NotImplementedError

    # ---- public API ----
    def set_processor(self, func: Optional[Callable[[FramePacket], FramePacket]]):
        """
        Set a processor callable. It receives and should return a FramePacket.
        Use to insert post-processing before the UI renders (e.g., undistort,
        draw overlays, inference heatmaps, etc.).
        """
        self._processor = func

    # ---- internal plumbing ----
    def _start_thread(self):
        """ def _start_thread: Describe purpose. """
        if self._thread is not None:
            return
        self._thread = self._make_thread()
        # Connect signals
        self._thread.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self._thread.fps_updated.connect(self._on_fps, Qt.QueuedConnection)
        self._thread.error_signal.connect(self._on_error, Qt.QueuedConnection)
        self._thread.start()

    def _stop_thread(self):
        """ def _stop_thread: Describe purpose. """
        if self._thread is not None:
            try:
                self._thread.stop()
            except Exception:
                pass
            self._thread = None

    def _on_frame(self, packet: FramePacket):
        """ def _on_frame: Describe purpose. """
        # Store latest only (UI will consume at throttled rate)
        if self._processor is not None:
            try:
                packet = self._processor(packet)
            except Exception as e:
                # Non-fatal; show warning once on label
                self.image_label.setText(f"Processor error: {e}")
        self._latest_packet = packet

    def _on_fps(self, fps: float):
        """ def _on_fps: Describe purpose. """
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str):
        """ def _on_error: Describe purpose. """
        self.image_label.setText(msg)
        self.stream_btn.setChecked(False)

    def _paint_latest(self):
        """ def _paint_latest: Describe purpose. """
        if self._latest_packet is None:
            return
        frame = self._latest_packet.frame
        if frame is None:
            return
        # Convert and zoom
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        sw = max(0.05, int(w * self.zoom_scale))
        sh = max(0.05, int(h * self.zoom_scale))
        resized = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0],
                      resized.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def stop_camera_on_close(self):
        """ def stop_camera_on_close: Handles UVC camera capture/streaming. """
        self._stop_thread()
class _BaseCameraWidget2(QWidget):
    """
    Keeps the same look-and-feel (label, image, fps, +/- zoom, Start/Stop),
    but rendering is throttled via a QTimer at TARGET_UI_FPS. Acquisition
    can run at full rate on its own thread.

    You can inject a frame processor with set_processor(callable).
    """
    def __init__(self, label_text: str):
        """ def __init__: Describe purpose. """
        super().__init__()
        self.label_text = label_text
        self.zoom_scale = 0.2
        self._latest_packet: Optional[FramePacket] = None
        self._processor: Optional[Callable[[FramePacket], FramePacket]] = None
        self._timer: Optional[QTimer] = None
        self._thread: Optional[QThread] = None
        self._build_ui()
        self._setup_paint_timer()

    def _build_ui(self):
        """ def _build_ui: Defines a UI widget/window. """
        layout = QVBoxLayout()

        self.label = QLabel(self.label_text)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.image_label = QLabel("No Frame")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(160, 120)
        self.image_label.setMaximumSize(CAMERA_VIEWPORT_SIZE)
        self.image_label.setFrameShape(QFrame.Box)
        layout.addWidget(self.image_label)

        self.fps_label = QLabel("FPS: 0.0")
        layout.addWidget(self.fps_label)

        zoom_layout = QHBoxLayout()
        self.zoom_in_button  = QPushButton("+")
        self.zoom_out_button = QPushButton("-")
        self.zoom_in_button.clicked.connect(self._zoom_in)
        self.zoom_out_button.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_in_button)
        layout.addLayout(zoom_layout)

        self.stream_btn = QPushButton("Start Stream")
        self.stream_btn.setCheckable(True)
        self.stream_btn.toggled.connect(self._toggle_stream)
        layout.addWidget(self.stream_btn)
        self.stream_btn.setText("Stop Stream")
        self._start_thread()

        self.setLayout(layout)

    def _setup_paint_timer(self):
        """ def _setup_paint_timer: Describe purpose. """
        self._timer = QTimer(self)
        interval_ms = int(1000 / max(1, int(TARGET_UI_FPS)))
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._paint_latest)
        self._timer.start()

    def _zoom_in(self):
        """ def _zoom_in: Describe purpose. """
        self.zoom_scale = min(3.0, self.zoom_scale + 0.1)

    def _zoom_out(self):
        """ def _zoom_out: Describe purpose. """
        self.zoom_scale = max(0.1, self.zoom_scale - 0.1)

    def _toggle_stream(self, checked: bool):
        """ def _toggle_stream: Describe purpose. """
        if checked:
            self.stream_btn.setText("Stop Stream")
            self._start_thread()
        else:
            self.stream_btn.setText("Start Stream")
            self._stop_thread()

    # ---- hooks to be implemented by subclasses ----
    def _make_thread(self) -> QThread:
        """ def _make_thread: Describe purpose. """
        raise NotImplementedError

    # ---- public API ----
    def set_processor(self, func: Optional[Callable[[FramePacket], FramePacket]]):
        """
        Set a processor callable. It receives and should return a FramePacket.
        Use to insert post-processing before the UI renders (e.g., undistort,
        draw overlays, inference heatmaps, etc.).
        """
        self._processor = func

    # ---- internal plumbing ----
    def _start_thread(self):
        """ def _start_thread: Describe purpose. """
        if self._thread is not None:
            return
        self._thread = self._make_thread()
        # Connect signals
        self._thread.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self._thread.fps_updated.connect(self._on_fps, Qt.QueuedConnection)
        self._thread.error_signal.connect(self._on_error, Qt.QueuedConnection)
        self._thread.start()

    def _stop_thread(self):
        """ def _stop_thread: Describe purpose. """
        if self._thread is not None:
            try:
                self._thread.stop()
            except Exception:
                pass
            self._thread = None

    def _on_frame(self, packet: FramePacket):
        """ def _on_frame: Describe purpose. """
        # Store latest only (UI will consume at throttled rate)
        if self._processor is not None:
            try:
                packet = self._processor(packet)
            except Exception as e:
                # Non-fatal; show warning once on label
                self.image_label.setText(f"Processor error: {e}")
        self._latest_packet = packet

    def _on_fps(self, fps: float):
        """ def _on_fps: Describe purpose. """
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str):
        """ def _on_error: Describe purpose. """
        self.image_label.setText(msg)
        self.stream_btn.setChecked(False)

    def _paint_latest(self):
        """ def _paint_latest: Describe purpose. """
        if self._latest_packet is None:
            return
        frame = self._latest_packet.frame
        if frame is None:
            return
        # Convert and zoom
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        sw = max(0.05, int(w * self.zoom_scale))
        sh = max(0.05, int(h * self.zoom_scale))
        resized = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0],
                      resized.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def stop_camera_on_close(self):
        """ def stop_camera_on_close: Handles UVC camera capture/streaming. """
        self._stop_thread()


class UVC_CameraWidget(_BaseCameraWidget2):
    def __init__(self, label, index):
        self.index = index
        super().__init__(label)
    def _make_thread(self) -> QThread:
        return UVC_CameraThread(self.index, WIDTH, HEIGHT)

class BAS_CameraWidget(_BaseCameraWidget1):
    def __init__(self, label, index):
        self.index = index
        super().__init__(label)
    def _make_thread(self) -> QThread:
        return BAS_CameraThread(self.index, TC_WIDTH, TC_HEIGHT)

# -----------------------------
# Servo Modbus Panel (Qt widget)
# -----------------------------
class ModbusPollThread(QThread):
    """Background thread to poll live & triggered positions at 0.1s interval."""
    update_values = Signal(int, int)  # triggered_val, live_val
    connection_error = Signal(str)

    def __init__(self, client: ModbusTcpClient, lock: threading.Lock, poll_interval=0.5):
        super().__init__()
        self.client = client
        self.lock = lock
        self.poll_interval = poll_interval
        self._running = True

    def run(self):
        while self._running:
            try:
                # read triggered (D100 -> address 100) and live (HC202 -> address 202)
                with self.lock:
                    rr1 = self.client.read_holding_registers(100)
                    rr2 = self.client.read_holding_registers(104)
                if hasattr(rr1, "isError") and rr1.isError():
                    raise Exception(f"Read error D100: {rr1}")
                if hasattr(rr2, "isError") and rr2.isError():
                    raise Exception(f"Read error HC104: {rr2}")
                trig = rr1.registers[0]
                live = rr2.registers[0]
                self.update_values.emit(trig, live)
            except Exception as e:
                # Emit connection error once (UI will show message)
                self.connection_error.emit(str(e))
            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False
        self.wait(500)

class ServoControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        # Modbus client setup
        self.plc_ip = "192.168.1.5"
        self.plc_port = 502
        self.client = ModbusTcpClient(host=self.plc_ip, port=self.plc_port)
        self.client.connect()
        self.lock = threading.Lock()
        # start poll thread
        self.poll_thread = ModbusPollThread(self.client, self.lock, poll_interval=0.1)
        self.poll_thread.update_values.connect(self._on_poll_update, Qt.QueuedConnection)
        self.poll_thread.connection_error.connect(self._on_poll_error, Qt.QueuedConnection)
        self.poll_thread.start()

    def _build_ui(self):
        root_layout = QVBoxLayout()
        title = QLabel("<b>Servo / PLC Control</b>")
        root_layout.addWidget(title)

        # Target RPM
        grp_rpm = QGroupBox("Target RPM (D150)")
        form_rpm = QFormLayout()
        self.entry_rpm = QLineEdit()
        btn_set_rpm = QPushButton("Set RPM")
        btn_set_rpm.clicked.connect(self.on_set_rpm)
        form_rpm.addRow(self.entry_rpm, btn_set_rpm)
        grp_rpm.setLayout(form_rpm)
        root_layout.addWidget(grp_rpm)

        # Controls (momentary push buttons)
        grp_ctrl = QGroupBox("Controls (momentary)")
        vctrl = QVBoxLayout()
        self.btn_start = QPushButton("Start (M31)")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_start.pressed.connect(lambda: self._momentary_coil(31))
        vctrl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop (M30)")
        self.btn_stop.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_stop.pressed.connect(lambda: self._momentary_coil(30))
        vctrl.addWidget(self.btn_stop)

        self.btn_estop = QPushButton("E-Stop (M0)")
        self.btn_estop.setStyleSheet("background-color: red; color: white;")
        self.btn_estop.pressed.connect(lambda: self._momentary_coil(0))
        vctrl.addWidget(self.btn_estop)

        grp_ctrl.setLayout(vctrl)
        root_layout.addWidget(grp_ctrl)

        # Cam offsets (D6200, D6204, D6208, D6212)
        grp_cam = QGroupBox("Cam Offsets (D6200..)")
        fcam = QFormLayout()
        self.cam_entries = {}
        for i in range(1, 5):
            addr = 6200 + (i-1)*4
            le = QLineEdit()
            btn = QPushButton(f"Write Cam {i}")
            btn.clicked.connect(lambda _, c=i: self.on_write_cam_offset(c))
            row = QHBoxLayout()
            row.addWidget(le)
            row.addWidget(btn)
            w = QWidget()
            w.setLayout(row)
            fcam.addRow(f"Cam {i} (D{addr})", w)
            self.cam_entries[i] = (addr, le)
        grp_cam.setLayout(fcam)
        root_layout.addWidget(grp_cam)

        # Cam offsets Normal cam 1,2...
        grp_cam = QGroupBox("Normal Cam Offsets")
        fcam = QFormLayout()
        self.normal_cam_entries = {}
        for i in range(1, 7):
            addr = 6200 + (i-1)*4
            le = QLineEdit()
            btn = QPushButton(f"Write Cam {i}")
            btn.clicked.connect(lambda _, c=i: self.on_write_normal_cam_offset(c))
            row = QHBoxLayout()
            row.addWidget(le)
            row.addWidget(btn)
            w = QWidget()
            w.setLayout(row)
            fcam.addRow(f"Cam {i}", w)
            self.normal_cam_entries[i] = le
        grp_cam.setLayout(fcam)
        root_layout.addWidget(grp_cam)

        #Read positions
        grp_read = QGroupBox("Positions")
        fpos = QFormLayout()
        self.entry_triggered = QLineEdit()
        self.entry_triggered.setReadOnly(True)
        btn_read = QPushButton("Read Now")
        btn_read.clicked.connect(self.on_read_now)
        row_trig = QHBoxLayout()
        row_trig.addWidget(self.entry_triggered)
        row_trig.addWidget(btn_read)
        wtr = QWidget()
        wtr.setLayout(row_trig)
        fpos.addRow("Triggered Pos (D100):", wtr)

        self.entry_live = QLineEdit()
        self.entry_live.setReadOnly(True)
        fpos.addRow("Live Pos (HC104):", self.entry_live)
        grp_read.setLayout(fpos)
        root_layout.addWidget(grp_read)

        # Spacer to push things up
        root_layout.addItem(QSpacerItem(20, 40, QSP.Minimum, QSP.Expanding))
        self.setLayout(root_layout)

    # ---------------------------
    # Modbus helper wrappers (no unit/slave)
    # ---------------------------
    def _write_register(self, address: int, value: int):
        try:
            with self.lock:
                rr = self.client.write_register(address, int(value))
            if hasattr(rr, "isError") and rr.isError():
                raise Exception(f"Write error at {address}: {rr}")
            return True
        except Exception as e:
            self._show_error(str(e))
            return False

    def _read_register(self, address: int):
        try:
            with self.lock:
                rr = self.client.read_holding_registers(address)
            if hasattr(rr, "isError") and rr.isError():
                raise Exception(f"Read error at {address}: {rr}")
            return rr.registers[0]
        except Exception as e:
            self._show_error(str(e))
            return None

    def _write_coil(self, address: int, state: bool):
        try:
            with self.lock:
                rr = self.client.write_coil(address, bool(state))
            if hasattr(rr, "isError") and rr.isError():
                raise Exception(f"Coil write error at {address}: {rr}")
            return True
        except Exception as e:
            self._show_error(str(e))
            return False

    # ---------------------------
    # UI callbacks
    # ---------------------------
    def on_set_rpm(self):
        txt = self.entry_rpm.text().strip()
        if txt == "":
            return
        try:
            v = int(txt)
        except:
            self._show_error("RPM must be integer")
            return
        # D150 -> address 150
        self._write_register(150, v)

    def on_write_cam_offset(self, cam_idx: int):
        addr, le = self.cam_entries[cam_idx]
        txt = le.text().strip()
        if txt == "":
            return
        try:
            v = int(txt)
        except:
            self._show_error("Offset must be integer")
            return
        self._write_register(addr, v)

    def on_write_normal_cam_offset(self, cam_idx: int):
        le = self.normal_cam_entries[cam_idx]
        txt = le.text().strip()
        if txt == "":
            return
        try:
            v = int(txt)
        except:
            self._show_error("Offset must be integer")
            return
        normal_cam_offset_value_write(cam_idx, v)

    def _momentary_coil(self, coil_addr: int, pulse_ms: int = 200):
        # Write ON, schedule OFF after pulse_ms
        ok = self._write_coil(coil_addr, True)
        if not ok:
            return
        # schedule off using QTimer to avoid blocking
        QTimer.singleShot(pulse_ms, lambda: self._write_coil(coil_addr, False))

    def on_read_now(self):
        trig = self._read_register(100)   # D100 -> 100
        live = self._read_register(104)   # HC202 -> 202
        if trig is not None:
            self.entry_triggered.setText(str(trig))
        if live is not None:
            self.entry_live.setText(str(live))

    def _on_poll_update(self, trig_val: int, live_val: int):
        # update UI fields from poll thread
        self.entry_triggered.setText(str(trig_val))
        self.entry_live.setText(str(live_val))

    def _on_poll_error(self, msg: str):
        # Show the error in the live field (non modal)
        self.entry_live.setText(f"ERR: {msg}")

    def _show_error(self, msg: str):
        # Non-blocking inline error display (we avoid messagebox spam)
        # show briefly in live entry
        self.entry_live.setText(f"ERR: {msg}")

    def stop(self):
        try:
            if hasattr(self, "poll_thread") and self.poll_thread is not None:
                self.poll_thread.stop()
                self.poll_thread = None
        except Exception:
            pass
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.close()
        except Exception:
            pass

# -----------------------------
# Main window (modified to add Servo panel on right side)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High FPS Multi-Camera Viewer + Servo Control")
        self.setMinimumSize(1600, 900)
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()

        left_grid  = QGridLayout()
        right_grid = QGridLayout()

        self.cameras = []

        # 6 UVC (left grid)
        for i in range(6):
            cam = UVC_CameraWidget(f"UVC Camera {i+1}", i)
            self.cameras.append(cam)
            r, c = divmod(i, 2)
            left_grid.addWidget(cam, r, c)

        # 4 Basler (middle/right grid)
        for i in range(4):
            cam = BAS_CameraWidget(f"Basler Camera {i+1}", i)
            self.cameras.append(cam)
            r, c = divmod(i, 2)
            right_grid.addWidget(cam, r, c)

        # Add camera grids to main layout (preserve original looking layout)
        main_layout.addLayout(left_grid, 2)
        main_layout.addLayout(right_grid, 2)

        # Servo control panel on the far right (new column)
        self.servo_panel = ServoControlPanel()
        main_layout.addWidget(self.servo_panel, 1)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def closeEvent(self, event):
        # stop camera threads
        for cam in self.cameras:
            cam.stop_camera_on_close()
        # stop servo panel
        try:
            self.servo_panel.stop()
        except Exception:
            pass
        super().closeEvent(event)

# -----------------------------
# Example: optional processor hook (kept same)
# -----------------------------
def example_processor(packet: FramePacket) -> FramePacket:
    frame = packet.frame
    text = time.strftime("%H:%M:%S", time.localtime(packet.ts))
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return packet

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    w = MainWindow()

    # (Optional) set processors if you like:
    # w.cameras[0].set_processor(example_processor)
    # w.cameras[6].set_processor(example_processor)

    w.show()
    sys.exit(app.exec())