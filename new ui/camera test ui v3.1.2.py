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
    QLineEdit, QGroupBox, QFormLayout, QSpacerItem, QSizePolicy as QSP, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView, QCheckBox
)
from PySide6.QtGui import QImage, QPixmap, QBrush, QColor, QFont
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

normal_camera_image_folder = "brush_knowledge/normal_camera_images"
tele_camera_image_folder = "brush_knowledge/tele_camera_images"

#model = YOLO(model_path)
#class_names = model.names
# UI paint throttle (frames-per-second)
TARGET_UI_FPS = 30   # set to 24 or 30 as you like

# -----------------------------
# Camera UI code (unchanged in behaviour / layout)
# -----------------------------

file_lock = threading.Lock()  # global lock for file access

plc_ip = "192.168.1.5"
normal_camera_index_list = [0,1,2,3,4,5]
tele_camera_index_list = [0,1,2,3]
normal_cam_offset_value=[]
tele_cam_offset_value=[]
tele_camera_wait_for_frame = 25000
#tele_camera_wait_for_frame = 100
take_picture = False

folder_brush_no = None
trigger=True # tele centric
        

with open("configuration.json", "r") as file:
    json_data = json.load(file)

normal_camera_index_list = json_data["normal_camera_index_list"]
tele_camera_index_list = json_data["tele_camera_index_list"]

NORMAL_CAM_FILE_PATH = "normal_camera_data.txt"
TELE_CAM_FILE_PATH = "tele_camera_data.txt"

def normal_cam_offset_value_write(cam_index, value):
    cam_offset_value_write(NORMAL_CAM_FILE_PATH, "Normal_cam:"+str(cam_index), value)

def tele_cam_offset_value_write(cam_index, value):
    cam_offset_value_write(TELE_CAM_FILE_PATH ,"Tele_cam:"+str(cam_index), value)
    
def cam_offset_value_write(FILE_PATH, cam_index, value):
    """Write or update a camera index and its value in the file."""
    if file_lock:
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
    global normal_cam_offset_value
    if len(normal_cam_offset_value) < 6:
        normal_cam_offset_value=[]
        for i in range(1,7):
            normal_cam_offset_value.append(normal_cam_offset_value_read_from_file("Normal_cam:"+str(i)))
    return normal_cam_offset_value[cam_index]

def tele_cam_offset_value_read(cam_index):
    global tele_cam_offset_value
    if len(tele_cam_offset_value) < 4:
        tele_cam_offset_value=[]
        for i in range(1,5):
            tele_cam_offset_value.append(tele_cam_offset_value_read_from_file("Tele_cam:"+str(i)))
    return tele_cam_offset_value[cam_index]
    
def normal_cam_offset_value_read_from_file(cam_index):
    """Read a camera index value from the file."""
    if file_lock:
        if not os.path.exists(NORMAL_CAM_FILE_PATH):
            return None

        with open(NORMAL_CAM_FILE_PATH, "r") as f:
            for line in f:
                parts = line.strip().split("=", 1)
                if len(parts) == 2 and parts[0] == str(cam_index):
                    return parts[1]
        return None

def tele_cam_offset_value_read_from_file(cam_index):
    """Read a camera index value from the file."""
    if file_lock:
        if not os.path.exists(TELE_CAM_FILE_PATH):
            return None

        with open(TELE_CAM_FILE_PATH, "r") as f:
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
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self.error_signal.emit(f"UVC {self.index}: failed to open")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -10)
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
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                time.sleep(2.5)
            ok, frame = self._cap.read()
            if not ok:
                self.error_signal.emit(f"UVC {self.index}: read failed")
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
            if ok:
                self.frame_ready.emit(FramePacket(frame))
                self._frame=frame
                frames += 1
                now = time.time()
                if now - last >= 1.0:
                    self.fps_updated.emit(frames / (now - last))
                    frames = 0
                    last = now
                
            time.sleep(0.05)

        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def capture_image(self):
        if(self._running):
           frame = self._frame
           return frame
 
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
        self._frame = None
        self.frame_no = 0
        self.frame_previous_no = 0
        
    def run(self):

        try:
            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            if len(devices) <= self.index:
                self.error_signal.emit(f"Basler {self.index}: device not found")
                return

            self._camera = pylon.InstantCamera(tlf.CreateDevice(devices[self.index]))
            self._camera.Open()
            if trigger:
                self._camera.TriggerSelector.SetValue('FrameStart')
                self._camera.TriggerMode.SetValue('On')
                self._camera.TriggerSource.SetValue('Line1')

            self._camera.ExposureAuto.SetValue('Off')
            self._camera.ExposureTime.SetValue(200)
            self._camera.GainAuto.SetValue('Off')
            self._camera.Gain.SetValue(30.0)
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
                grab = self._camera.RetrieveResult(tele_camera_wait_for_frame, pylon.TimeoutHandling_Return)
                if grab.IsValid():
                    img = converter.Convert(grab)
                    frame = img.GetArray()
                    self._frame = frame
                    self.frame_no +=1 
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

    def capture_image(self):
        if(self._running):
            while(self.frame_no == self.frame_previous_no):
                pass
            self.frame_previous_no = self.frame_no
            if self._frame is not None:
                frame = self._frame
                return frame
            else:
                print("Tele Cam frame not available")
                return None

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
        self.index = normal_camera_index_list[index]
        super().__init__(label)
    
    def _make_thread(self) -> QThread:
        self.cameraThread = UVC_CameraThread(self.index, WIDTH, HEIGHT)
        return self.cameraThread
    
    def getCameraThread(self):
        return self.cameraThread
        
class BAS_CameraWidget(_BaseCameraWidget1):
    def __init__(self, label, index):
        self.index = tele_camera_index_list[index]
        super().__init__(label)
    
    def _make_thread(self) -> QThread:
        self.cameraThread = BAS_CameraThread(self.index, TC_WIDTH, TC_HEIGHT)
        return self.cameraThread 
    
    def getCameraThread(self):
        return self.cameraThread
    
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
                trig = self._read_register_32_bit(100)
                live = self._read_register_32_bit(104)
                self.update_values.emit(trig, live)
            except Exception as e:
                # Emit connection error once (UI will show message)
                self.connection_error.emit(str(e))
            time.sleep(self.poll_interval)
    
    def _read_register_32_bit(self, address: int):
        try:
            rr = 0
            with self.lock:
                rr1 = self.client.read_holding_registers(address)
                if hasattr(rr1, "isError") and rr1.isError():
                    raise Exception(f"Read error at {address}: {rr1}")
                rr1 = rr1.registers[0]

                rr2 = self.client.read_holding_registers(address+1)
                if hasattr(rr2, "isError") and rr2.isError():
                    raise Exception(f"Read error at {address}: {rr2}")
                rr2 = rr2.registers[0]
                
                rr = (rr2 << 16) | rr1
                # rr = (rr1 << 16) | rr2
                # print("32-bit value:",sensorTriggeredPosition)
                if rr >= 0x80000000:  # if MSB is 1
                    rr -= 0x100000000
            return rr
        except Exception as e:
            self._show_error(str(e))
            return None
        
    def stop(self):
        self._running = False
        self.wait(500)

class ServoControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        # Modbus client setup
        self.plc_ip = plc_ip
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
        title = QLabel("<b>Main Settings</b>")
        root_layout.addWidget(title)

        # Controls (momentary push buttons)
        grp_ctrl = QGroupBox("Servo Controls")
        main_layout = QVBoxLayout()
        vctrl = QVBoxLayout()
        self.btn_start = QPushButton("Start (M31)")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_start.pressed.connect(lambda: self._momentary_coil(31))
        vctrl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop (M30)")
        # self.btn_stop.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_stop.setStyleSheet("background-color: red; color: white;")
        self.btn_stop.pressed.connect(lambda: self._momentary_coil(30))
        vctrl.addWidget(self.btn_stop)

        # self.btn_estop = QPushButton("E-Stop (M0)")
        # self.btn_estop.setStyleSheet("background-color: red; color: white;")
        # self.btn_estop.pressed.connect(lambda: self._momentary_coil(0))
        # vctrl.addWidget(self.btn_estop)
        main_layout.addLayout(vctrl)

        # Target RPM
        form_rpm = QFormLayout()
        self.entry_rpm = QLineEdit()
        btn_set_rpm = QPushButton("Set RPM")
        btn_set_rpm.clicked.connect(self.on_set_rpm)
        row_trig = QHBoxLayout()
        row_trig.addWidget(self.entry_rpm)
        row_trig.addWidget(btn_set_rpm)
        wtr = QWidget()
        wtr.setLayout(row_trig)
        form_rpm.addRow("Target RPM (D150):", wtr)
        main_layout.addLayout(form_rpm)

        grp_ctrl.setLayout(main_layout)
        root_layout.addWidget(grp_ctrl)

        #Read positions
        grp_read = QGroupBox("Disc plate Servo positions")
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

        # #Light control
        # grp_ctrl = QGroupBox("Light Controls")
        # main_layout = QVBoxLayout()

        # # --- Normal Light ---
        # self.btn_normal_light = QPushButton("Turn ON Normal Light (M3)")
        # self.btn_normal_light.setCheckable(True)
        # self.btn_normal_light.setStyleSheet("background-color: #4CAF50; color: white;")
        # self.btn_normal_light.clicked.connect(lambda: self.toggle_light(self.btn_normal_light, 3))
        # main_layout.addWidget(self.btn_normal_light)

        # # --- Telecentric Light ---
        # self.btn_tele_light = QPushButton("Turn ON Telecentric Light (M2)")
        # self.btn_tele_light.setCheckable(True)
        # self.btn_tele_light.setStyleSheet("background-color: #4CAF50; color: white;")
        # self.btn_tele_light.clicked.connect(lambda: self.toggle_light(self.btn_tele_light, 2))
        # main_layout.addWidget(self.btn_tele_light)

        # grp_ctrl.setLayout(main_layout)
        # root_layout.addWidget(grp_ctrl)


        grp_ctrl = QGroupBox("Light Controls")
        main_layout = QVBoxLayout()

        # --- Normal Light Switch (M3) ---
        self.switch_normal = QCheckBox("Normal Light (M3)")
        self.switch_normal.setStyleSheet(self.switch_style())
        self.switch_normal.stateChanged.connect(lambda state: self.toggle_light(3, state))
        main_layout.addWidget(self.switch_normal)

        # --- Telecentric Light Switch (M4) ---
        self.switch_tele = QCheckBox("Tele Centric Light (M2)")
        self.switch_tele.setStyleSheet(self.switch_style())
        self.switch_tele.stateChanged.connect(lambda state: self.toggle_light(2, state))
        main_layout.addWidget(self.switch_tele)

        grp_ctrl.setLayout(main_layout)
        root_layout.addWidget(grp_ctrl)

        # Cam offsets (D6200, D6204, D6208, D6212)
        # Cam offsets (D20200, D20204, D20208, D20212)
        grp_cam = QGroupBox("Cam Offsets (D20200..)")
        fcam = QFormLayout()
        self.cam_entries = {}
        for i in range(1, 5):
            addr = 20200 + (i-1)*4
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
        
    def _read_register_32_bit(self, address: int):
        try:
            rr = 0
            with self.lock:
                rr1 = self.client.read_holding_registers(address)
                if hasattr(rr1, "isError") and rr1.isError():
                    raise Exception(f"Read error at {address}: {rr1}")
                rr1 = rr1.registers[0]

                rr2 = self.client.read_holding_registers(address+1)
                if hasattr(rr2, "isError") and rr2.isError():
                    raise Exception(f"Read error at {address}: {rr2}")
                rr2 = rr2.registers[0]
                
                rr = (rr2 << 16) | rr1
                # rr = (rr1 << 16) | rr2
                # print("32-bit value:",sensorTriggeredPosition)
                if rr >= 0x80000000:  # if MSB is 1
                    rr -= 0x100000000
            return rr
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
        
    def _read_coil(self, address: int):
        try:
            with self.lock:
                rr = self.client.read_coils(address)
            if hasattr(rr, "isError") and rr.isError():
                raise Exception(f"Read error at {address}: {rr}")
            return rr.bits[0]
        except Exception as e:
            self._show_error(str(e))
            return None

    # def toggle_light(self, button, coil_addr):
    #     """Toggle light ON/OFF based on current coil state."""
    #     try:
    #         # Read current coil state
    #         state = self._read_coil(coil_addr)
    #         if state is None:
    #             return  # read failed
            
    #         # Toggle the state
    #         new_state = not state
    #         self._write_coil(coil_addr, new_state)

    #         # Update button appearance and text
    #         if new_state:
    #             button.setText(f"Turn OFF {'Normal' if coil_addr==3 else 'Telecentric'} Light (M{coil_addr})")
    #             button.setStyleSheet("background-color: #FF5252; color: white;")  # red for ON
    #         else:
    #             button.setText(f"Turn ON {'Normal' if coil_addr==3 else 'Telecentric'} Light (M{coil_addr})")
    #             button.setStyleSheet("background-color: #4CAF50; color: white;")  # green for OFF

    #     except Exception as e:
    #         print(f"Error toggling light M{coil_addr}: {e}")

    def toggle_light(self, coil_no, state):
        # is_on = state == Qt.Checked
        is_on = state
        print(f"Coil {coil_no} {'ON' if is_on else 'OFF'}")
        self._write_coil(coil_no, is_on)
        print(f"Setting coil {coil_no} to {is_on}")

    def switch_style(self):
        return """
        QCheckBox::indicator {
            width: 50px;
            height: 25px;
        }
        QCheckBox::indicator:unchecked {
            border-radius: 12px;
            background-color: #ccc;
        }
        QCheckBox::indicator:checked {
            border-radius: 12px;
            background-color: #4CAF50;
        }
        QCheckBox {
            font-size: 14px;
            padding: 6px;
        }
        """
    

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
            self._show_error("empty text")
            return
        try:
            v = int(txt)
        except:
            self._show_error("Offset must be integer")
            return
        self._write_register(addr, v)
        tele_cam_offset_value_write(cam_idx, v)

    def on_write_normal_cam_offset(self, cam_idx: int):
        le = self.normal_cam_entries[cam_idx]
        txt = le.text().strip()
        if txt == "":
            self._show_error("empty text")
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
        trig = self._read_register_32_bit(100)   # D100 -> 100
        live = self._read_register_32_bit(104)   # HC202 -> 202
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

class BrushStatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("<b>Brush Current Status & History</b>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create Table
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setRowCount(0)
        # headers = ["ID"] + [f"{i}" for i in range(1, 7)] + [f"{i}" for i in range(1, 5)]
        # self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        # self.header_table.setFixedHeight(60)
        # self.header_table.setFocusPolicy(Qt.NoFocus)
        # self.header_table.setSelectionMode(QTableWidget.NoSelection)

         # --- Create fake two-level header using the first row ---
        top_headers = ["ID", "Normal Camera", "", "", "", "", "", "Tele Camera", "", "", ""]
        sub_headers = ["Brush"] + [f"{i}" for i in range(1, 7)] + [f"{i}" for i in range(1, 5)]

        # Set the actual header row to sub headers
        self.table.setHorizontalHeaderLabels(sub_headers)

        # --- Merge the top header visually ---
        self.table.insertRow(0)
        for col, text in enumerate(top_headers):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            #item.setBackground(QColor(230, 230, 230))
            item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.table.setItem(0, col, item)

        # Merge cells manually for group labels
        self.table.setSpan(0, 0, 1, 1)  # Brush ID
        self.table.setSpan(0, 1, 1, 6)  # Normal Camera
        self.table.setSpan(0, 7, 1, 4)  # Tele Camera

        # Optional: resize columns evenly
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # ✅ Set column widths
        self.table.setColumnWidth(0, 35)  # BrushID column
        for i in range(1, 11):
            self.table.setColumnWidth(i, 55)  # Camera columns
        layout.addWidget(self.table)

        self.setLayout(layout)

        # # Fill with sample data (for testing)
        # for row in range(5):
        #     self.add_brush_result(row + 1, [("Good" if (row + i) % 2 == 0 else "Bad") for i in range(1, 11)])

    def add_brush_result(self, brush_id: int, camera_results: list[str]):
        """Add one brush record to the table."""
        # Remove oldest row if 20 rows already exist
        if self.table.rowCount() >= 20:
            self.table.removeRow(0)  # removes the first (oldest) row

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(brush_id)))
        for i, result in enumerate(camera_results, start=1):
            item = QTableWidgetItem(result)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, i, item)

    def update_brush_status(self, brush_id: int, camera_no: int, status: str):
        """
        Update the status of a specific camera for a given brush ID.
        camera_no starts from 1 (since column 0 is BrushID).
        """
        target_id = str(brush_id)

        # Loop through all rows to find the brush ID
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == target_id:
                # Found the brush; now update camera status
                if 1 <= camera_no <= 10:
                    new_item = QTableWidgetItem(status)
                    new_item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, camera_no, new_item)
                else:
                    print(f"Invalid camera number: {camera_no}")
                return True

        print(f"Brush ID {brush_id} not found.")
        return False

    def remove_brush_by_id(self, brush_id: int):
        """Remove a brush record from the table by its Brush ID."""
        target_id = str(brush_id)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == target_id:
                self.table.removeRow(row)
                return True  # Successfully removed
        return False  # Brush ID not found
    

class TrainingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        root_layout = QVBoxLayout()
        title = QLabel("<b>Training</b>")
        root_layout.addWidget(title)

        # --- Start Button ---
        grp_ctrl = QGroupBox("Yolo Training Controls")
        
        layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.on_start_training)
        layout.addWidget(self.btn_start)

        # --- Stop Button ---
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.on_stop_training)
        layout.addWidget(self.btn_stop)

        # layout.addStretch()
        # self.setLayout(layout)
        grp_ctrl.setLayout(layout)
        root_layout.addWidget(grp_ctrl)

        # Spacer to push things up
        root_layout.addItem(QSpacerItem(20, 40, QSP.Minimum, QSP.Expanding))
        self.setLayout(root_layout)

    # --- Methods called when buttons are pressed ---
    def on_start_training(self):
        global take_picture
        if not take_picture:
            print("Training started")
            take_picture = True
            self.create_folder_for_normal_cameras()
            self.create_folder_for_tele_cameras()
        # TODO: add your start logic here (e.g. self.parent().start_training_process())

    def on_stop_training(self):
        global take_picture
        if take_picture:
            print("Training stopped")
            take_picture = False


    def create_folder_for_normal_cameras(self):
        self.create_folder_for_cameras(normal_camera_image_folder, 6)
    
    def create_folder_for_tele_cameras(self):
        self.create_folder_for_cameras(tele_camera_image_folder, 4)

    def create_folder_for_cameras(self, base_path, index):
        for(cam_idx) in range(1, index+1):
            cam_folder_path = os.path.join(base_path, f"Camera_{cam_idx}")
            self.create_folder_if_not_exists(cam_folder_path)
            
    def create_folder_if_not_exists(self, folder_path):
        global folder_brush_no

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Base path
        base_path = folder_path
        # Get list of folders inside base_path
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        # Filter numeric folder names only
        numbers = [int(f) for f in subfolders if f.isdigit()]
        # Find next folder number
        next_number = max(numbers) + 1 if numbers else 1
        # Build new folder path
        new_folder = os.path.join(base_path, str(next_number))
        # Create the new folder
        os.makedirs(new_folder, exist_ok=True)
        folder_brush_no = next_number
        
        print(f"✅ Created folder: {new_folder}")

# -----------------------------
# Main window (modified to add Servo panel on right side)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High FPS Multi-Camera Viewer + Servo Control")
        self.setMinimumSize(1600, 900)
        self._init_ui()
        self.mainloop = threading.Thread(target=self.mainloop)
        self.mainloop.start()

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()

        left_grid  = QGridLayout()
        right_grid = QGridLayout()

        self.normal_cameras = []
        self.tele_cameras = []

        # 6 UVC (left grid)
        for i in range(6):
            cam = UVC_CameraWidget(f"UVC Camera {i+1}", i)
            self.normal_cameras.append(cam)
            r, c = divmod(i, 2)
            left_grid.addWidget(cam, r, c)

        # 4 Basler (middle/right grid)
        for i in range(4):
            cam = BAS_CameraWidget(f"Basler Camera {i+1}", i)
            self.tele_cameras.append(cam)
            r, c = divmod(i, 2)
            right_grid.addWidget(cam, r, c)

        # Add camera grids to main layout (preserve original looking layout)
        main_layout.addLayout(left_grid, 2)
        main_layout.addLayout(right_grid, 2)

        # -------------------------------
        # RIGHT PANEL - TabWidget (Status + Settings)
        # -------------------------------

        # Tab widget for right-side panel
        self.right_tabs = QTabWidget()

        # Create brush status and servo panels
        self.brush_status_panel = BrushStatusPanel()
        self.servo_panel = ServoControlPanel()
        self.training_panel = TrainingPanel()  # <-- new panel

        # Add both as tabs
        self.right_tabs.addTab(self.brush_status_panel, "Brush Status")
        self.right_tabs.addTab(self.servo_panel, "Settings")
        self.right_tabs.addTab(self.training_panel, "Training")  # <-- add new tab

        # Default show Brush Status
        self.right_tabs.setCurrentIndex(0)

        # Add tab widget to main layout
        main_layout.addWidget(self.right_tabs, 2)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def mainloop(self):
        brushContainer = []
        brushId = 0
        while True:
            isBrushDetected = self.servo_panel._read_coil(4)
            #print(isBrushDetected)
            if(isBrushDetected):
                print("Brush Triggered")
                brushId+=1
                self.brush_status_panel.add_brush_result(brushId, ["Pending"]*10)
                time.sleep(0.1)
                sensorTriggeredPosition = self.servo_panel._read_register_32_bit(100)
                #livePosition = int(self.servo_panel._read_register(104))
                #print("Live Position: "+str(livePosition))
                print("Sensor Triggered Position:"+str(sensorTriggeredPosition))
                brushThread = BrushThread(brushId, sensorTriggeredPosition, self.normal_cameras, self.tele_cameras, self.servo_panel, self.brush_status_panel)
                brushContainer.append(brushThread)
                while(isBrushDetected):
                    isBrushDetected = self.servo_panel._read_coil(4)
            time.sleep(0.1)
        
    def closeEvent(self, event):
        # stop camera threads
        for cam in self.normal_cameras:
            cam.stop_camera_on_close()
        for cam in self.tele_cameras:
            cam.stop_camera_on_close()
        # stop servo panel
        try:
            self.servo_panel.stop()
        except Exception:
            pass
        super().closeEvent(event)

class BrushThread():
    def __init__(self, brushID: int, sensorTriggeredPosition, normal_cameras, tele_cameras, servo_panel: ServoControlPanel, brush_status_panel: BrushStatusPanel):
        self.brushID = brushID
        self.sensorTriggeredPosition = sensorTriggeredPosition
        self.normal_cameras = normal_cameras 
        self.tele_cameras = tele_cameras 
        self.servo_panel = servo_panel
        self.brush_status_panel = brush_status_panel
        self.delay = 0.050
        self.tolarance = 400
        self.run()

    def run(self):
        self.run_normal_cam_thread = threading.Thread(target=self.run_normal_camera)
        self.run_normal_cam_thread.start()
        self.run_tele_cam_thread = threading.Thread(target=self.run_tele_camera)
        self.run_tele_cam_thread.start()
        
    def run_normal_camera(self):
        normal_camere_future_positions = []
        # for cam in self.normal_cameras:
        #     #print(cam.index)
        #     normal_cam_position = normal_cam_offset_value_read(cam.index)
        #     if normal_cam_position:
        #         normal_camere_future_positions.append(int(normal_cam_position)+self.sensorTriggeredPosition)
        #     else:
        #         normal_camere_future_positions.append(0)

        for index in range(0,6):
            #print(cam.index)
            normal_cam_position = normal_cam_offset_value_read(index)
            if normal_cam_position:
                normal_camere_future_positions.append(int(normal_cam_position)+self.sensorTriggeredPosition)
            else:
                normal_camere_future_positions.append(0)

        normal_cam_index=0
        # print("Normal Cam positions: ")
        # print(normal_camere_future_positions)
        
        while(normal_cam_index < len(normal_camere_future_positions)):  
            #start_time = time.time()  
            livePosition = int(self.servo_panel._read_register(104))
            #end_time = time.time()
            #print(f"⏱️ Time taken: {end_time - start_time:.4f} seconds")
            #print("Live Position: "+str(livePosition))
            #print(normal_camere_future_positions)
            #print("Normal Cam future position: "+str(normal_camere_future_positions[normal_cam_index]))
            if(livePosition is not None and normal_camere_future_positions[normal_cam_index]-self.tolarance <= livePosition):
                print(f"Capturing images for Brush ID: {self.brushID} at position {livePosition} for Normal Camera {normal_cam_index+1}")
                # Update brush 101, camera 3 → "OK"
                self.brush_status_panel.update_brush_status(self.brushID, normal_cam_index+1, "Good")
                self.read_normal_camera_image(self.brushID, normal_cam_index+1, self.normal_cameras[normal_cam_index].getCameraThread().capture_image())
                normal_cam_index+=1
            elif(livePosition is not None and livePosition >= normal_camere_future_positions[normal_cam_index]+self.tolarance):
                print(f"Out off range - Not Capturing images for Brush ID: {self.brushID} at position {livePosition} for Normal Camera {normal_cam_index+1}")
                self.brush_status_panel.update_brush_status(self.brushID, normal_cam_index+1, "Not Captured")
                normal_cam_index+=1

            time.sleep(self.delay)
            # time.sleep(2)
        print("Normal camera closed")

    def run_tele_camera(self):
        tele_camera_future_positions = []

        # for cam in self.tele_cameras:
        #     tele_cam_position = tele_cam_offset_value_read(cam.index)
        #     if tele_cam_position:
        #         tele_camera_future_positions.append(int(tele_cam_position)+self.sensorTriggeredPosition)
        #     else:
        #         tele_camera_future_positions.append(0)

        for index in range(0,4):
            tele_cam_position = tele_cam_offset_value_read(index)
            if tele_cam_position:
                tele_camera_future_positions.append(int(tele_cam_position)+self.sensorTriggeredPosition)
            else:
                tele_camera_future_positions.append(0)

        tele_cam_index=0
        # print("Normal Cam positions: ")
        # print(normal_camere_future_positions)
        
        while(tele_cam_index < len(tele_camera_future_positions)):  
            #start_time = time.time()
            livePosition = int(self.servo_panel._read_register(104))
            #end_time = time.time()
            #print(f"⏱️ Time taken: {end_time - start_time:.4f} seconds")
            #print("Live Position: "+str(livePosition))
            #print(normal_camere_future_positions)
            #print("Normal Cam future position: "+str(normal_camere_future_positions[normal_cam_index]))
            if(livePosition is not None and tele_camera_future_positions[tele_cam_index]-self.tolarance <= livePosition):
                print(f"Capturing images for Brush ID: {self.brushID} at position {livePosition} for Tele Camera {tele_cam_index+1}")
                self.brush_status_panel.update_brush_status(self.brushID, 6 + tele_cam_index + 1, "Good")
                tele_image = self.tele_cameras[tele_cam_index].getCameraThread().capture_image()
                if tele_image is not None:
                    self.read_tele_camera_image(self.brushID, tele_cam_index+1, tele_image)
                else:
                    print(f"Not Capturing images for Brush ID: {self.brushID} at position {livePosition} for Tele Camera {tele_cam_index+1}")
                tele_cam_index+=1
            elif(livePosition is not None and livePosition >= tele_camera_future_positions[tele_cam_index]+self.tolarance):
                print(f"Out off range - Capturing images for Brush ID: {self.brushID} at position {livePosition} for Tele Camera {tele_cam_index+1}")
                self.brush_status_panel.update_brush_status(self.brushID, 6 + tele_cam_index + 1, "Not Captured")
                tele_cam_index+=1

            time.sleep(self.delay)
            # time.sleep(2)
        print("Tele camera closed")

    def read_normal_camera_image(self, brushID, cameraId, frame):
        if take_picture:
            folder_path = os.path.join(normal_camera_image_folder, f"Camera_{cameraId}")
            folder_path = os.path.join(folder_path, str(folder_brush_no))

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            timestamp = int(time.time() * 1000)
            filename = os.path.join(folder_path, f"normal_cam_{cameraId}_brush_{brushID}_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved Normal Camera image: {filename}")
    
    def read_tele_camera_image(self, brushID, cameraId, frame):
        if take_picture:
            folder_path = os.path.join(tele_camera_image_folder, f"Camera_{cameraId}")
            folder_path = os.path.join(folder_path, str(folder_brush_no))

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            timestamp = int(time.time() * 1000)
            filename = os.path.join(folder_path, f"tele_cam_{cameraId}_brush_{brushID}_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved Normal Camera image: {filename}")

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