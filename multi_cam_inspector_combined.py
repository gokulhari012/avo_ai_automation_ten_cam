#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera Inspector (Combined)
- Preserves original UI layout: 6 UVC (left grid) + 4 Basler (right grid)
- UVC cameras: motion gating + YOLO detection (from your original)
- Basler cameras: live subpixel measurements using template data produced
  by your Template Maker (./templates/template_1.json + template_1.bmp)
- Template matching: tries InvariantTM if available; otherwise falls back
  to OpenCV matchTemplate (+ optional horizontal flip) with downscaled search
- No extra UI controls are added to keep the same appearance. If no template
  files are found, Basler streams pass through unmodified.
"""
import sys, time, math, os
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
import cv2

# --- Optional invariant matching (best) ---
try:
    from InvariantTM import invariant_match_template  # optional dependency
    HAVE_INVARIANT_TM = True
except Exception:
    HAVE_INVARIANT_TM = False

# --- Pylon / Qt ---
from pypylon import pylon
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QGridLayout, QMainWindow, QFrame, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
import qdarkstyle

# --- YOLO (UVC only; set your model path below) ---
from ultralytics import YOLO

# ==================================================
# Config (tweak internals only; UI kept the same)
# ==================================================
CAMERA_VIEWPORT_SIZE = QSize(480, 320)
WIDTH  = 1920
HEIGHT = 1200

# Path to your trained YOLO model (unchanged from your original)
YOLO_MODEL_PATH = "C:/Users/Karthick-PC/runs/detect/train3/weights/best.pt"

# UI paint throttle (frames-per-second)
TARGET_UI_FPS = 30   # set to 24 or 30 as you like

# Template / measurement config
TEMPLATE_DIR = "./templates"
TEMPLATE_BMP = os.path.join(TEMPLATE_DIR, "template_1.bmp")
TEMPLATE_JSON = os.path.join(TEMPLATE_DIR, "template_1.json")
MM_PER_PIXEL_DEFAULT = 0.0075
TOLERANCE_DEFAULT = 0.01
PARALLEL_SAMPLES = 11   # number of adjacent scanlines/columns to average (odd)
DELTA_OFF = 50          # offset distance for reference profiles
EDGE_THRESH = 128
UI_SCALE = 1.5          # scale text/lines drawn on overlay

# ==================================================
# Lightweight frame packet used by threads
# ==================================================
class FramePacket:
    __slots__ = ("frame", "ts")
    def __init__(self, frame):
        self.frame = frame
        self.ts = time.time()

# ==================================================
# Utility: Subpixel edge detection (1D crossing)
# ==================================================
def detect_subpixel_edges(profile: np.ndarray, threshold: float = EDGE_THRESH):
    # profile: 1D float32/uint8 vector
    indices = np.where(np.diff(np.sign(profile - threshold)))[0]
    subpixel_edges = []
    for idx in indices:
        x0, x1 = idx, idx + 1
        y0, y1 = profile[x0], profile[x1]
        if y1 != y0:
            subpixel = x0 + (threshold - y0) / (y1 - y0)
            subpixel_edges.append(subpixel)
    return subpixel_edges

# ==================================================
# Template alignment + crop helpers
# ==================================================
def align_image_by_minarea_rect(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Roughly align object using minAreaRect on a downscaled binary mask."""
    ds = 7
    image_ds = cv2.resize(image_bgr, (image_bgr.shape[1] // ds, image_bgr.shape[0] // ds))
    gray = cv2.cvtColor(image_ds, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100000 // ds:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w > h:
            angle += 90
            w, h = h, w
        center = (int(cx) * ds, int(cy) * ds)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_bgr, rot_mat, (image_bgr.shape[1], image_bgr.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))
        return rotated, True
    return image_bgr, False

def invariant_or_fallback_match(img_rgb: np.ndarray, template_rgb: np.ndarray,
                                matched_thresh: float = 0.70) -> Optional[Tuple[int,int,int,int,float]]:
    """
    Returns (x, y, w, h, angle_deg) in source image where template best matches.
    If InvariantTM is available, use it with 90° steps; else fallback to cv2.matchTemplate.
    """
    ds = 25  # downscale for search
    img_ds = cv2.resize(img_rgb, (img_rgb.shape[1] // ds, img_rgb.shape[0] // ds))
    tpl_ds = cv2.resize(template_rgb, (template_rgb.shape[1] // ds, template_rgb.shape[0] // ds))

    if HAVE_INVARIANT_TM:
        points_list = invariant_match_template(
            rgbimage=img_ds,
            rgbtemplate=tpl_ds,
            method="TM_CCOEFF_NORMED",
            matched_thresh=matched_thresh,
            rot_range=[0, 360],
            rot_interval=90,
            scale_range=[99, 101],
            scale_interval=1,
            rm_redundant=True,
            minmax=True
        )
        if not points_list:
            # try horizontal flip template
            tpl_ds_flipped = cv2.flip(tpl_ds, 1)
            points_list = invariant_match_template(
                rgbimage=img_ds,
                rgbtemplate=tpl_ds_flipped,
                method="TM_CCOEFF_NORMED",
                matched_thresh=matched_thresh,
                rot_range=[0, 360],
                rot_interval=90,
                scale_range=[99, 101],
                scale_interval=1,
                rm_redundant=True,
                minmax=True
            )
        if not points_list:
            return None
        (px, py), angle, scale, _ = points_list[0]
        x = int(px * ds); y = int(py * ds)
        h, w = template_rgb.shape[:2]
        w_scaled = int(w * scale / 100); h_scaled = int(h * scale / 100)
        return x, y, w_scaled, h_scaled, angle

    # Fallback: cv2.matchTemplate with optional horizontal flip (no rotation/scale search)
    best_score = -1.0
    best = None
    for flipped in (False, True):
        tpl = cv2.flip(tpl_ds, 1) if flipped else tpl_ds
        res = cv2.matchTemplate(img_ds, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best = (max_loc, flipped)
    if best is None or best_score < matched_thresh:
        return None
    (px, py), flipped = best
    x = int(px * ds); y = int(py * ds)
    h, w = template_rgb.shape[:2]
    return x, y, w, h, 0.0

def match_and_crop_aligned(img_bgr: np.ndarray, template_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Align rough rotation, then match and crop around template region."""
    img_bgr_aligned, _ = align_image_by_minarea_rect(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr_aligned, cv2.COLOR_BGR2RGB)

    found = invariant_or_fallback_match(img_rgb, template_rgb)
    if not found:
        return img_bgr_aligned, False

    x, y, w, h, angle = found
    x_end = min(x + w, img_rgb.shape[1])
    y_end = min(y + h, img_rgb.shape[0])
    if x < 0 or y < 0 or x_end <= x or y_end <= y:
        return img_bgr_aligned, False

    crop = img_rgb[y:y_end, x:x_end]
    # De-rotate crop if needed
    if abs(angle) > 1e-3:
        center = (crop.shape[1] // 2, crop.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        crop = cv2.warpAffine(crop, rot_mat, (crop.shape[1], crop.shape[0]),
                              flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))
    # Return as BGR for downstream drawing
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), True

# ==================================================
# Measurement overlay (ported & streamlined)
# ==================================================
def _filter_mean(distances: list[float]) -> Optional[float]:
    if not distances:
        return None
    data = np.array(distances, dtype=np.float32)
    mean = float(np.mean(data))
    std  = float(np.std(data))
    if std > 0:
        z = np.abs((data - mean) / std)
        data = data[z < 2.0]
    if data.size == 0:
        return None
    return float(np.mean(data))

def analyze_frame_overlay(
    image_bgr: np.ndarray,
    template_data: Dict[str, Any],
    mm_per_pixel: float = MM_PER_PIXEL_DEFAULT,
    delta: int = DELTA_OFF,
    num_parallel: int = PARALLEL_SAMPLES,
    threshold: int = EDGE_THRESH,
    tolerance: float = TOLERANCE_DEFAULT,
    scale: float = UI_SCALE
) -> np.ndarray:
    """
    Draws measurements & deviation table on top of image_bgr (in place).
    Returns the same image reference for convenience.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    output = image_bgr

    h_vals, v_vals = [], []
    h_vals_def = template_data.get("h_vals_def", [])
    v_vals_def = template_data.get("v_vals_def", [])

    font_scale = 0.5 * scale
    font_thickness = int(1 * scale)
    line_thickness = int(1 * scale)
    spacing = int(18 * scale)
    table_bg_color = (220, 220, 220)

    half = num_parallel // 2

    # Horizontal lines
    for coords in template_data.get("horizontal_lines", []):
        y_center = int(coords[1][1])
        if delta + half >= y_center or y_center >= gray.shape[0] - (delta + half):
            continue
        all_distances = []
        for off in range(-half, half + 1):
            y = y_center + off
            prof     = gray[y, :]
            prof_top = gray[y - delta, :]
            prof_bot = gray[y + delta, :]
            edges    = detect_subpixel_edges(prof, threshold)
            e_top    = detect_subpixel_edges(prof_top, threshold)
            e_bot    = detect_subpixel_edges(prof_bot, threshold)
            for i in range(0, min(len(edges), len(e_top), len(e_bot)) - 1, 2):
                x0, x1   = edges[i], edges[i + 1]
                x0_t, x1_t = e_top[i], e_top[i + 1]
                x0_b, x1_b = e_bot[i], e_bot[i + 1]
                dx = ((x1_b - x1_t) + (x0_b - x0_t)) / 2.0
                dy = 2.0 * delta
                ang = math.atan2(dx, dy)
                dist_px = abs(x1 - x0) * abs(math.cos(ang))
                all_distances.append(dist_px * mm_per_pixel)
        m = _filter_mean(all_distances)
        if m is not None:
            h_vals.append(m)
            cv2.putText(output, f"{m:.2f} mm", (10, y_center - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
            cv2.line(output, (0, y_center), (output.shape[1], y_center),
                     (255, 255, 0), line_thickness)

    # Vertical lines
    for coords in template_data.get("vertical_lines", []):
        x_center = int(coords[0][0])
        if delta + half >= x_center or x_center >= gray.shape[1] - (delta + half):
            continue
        all_distances = []
        for off in range(-half, half + 1):
            x = x_center + off
            prof      = gray[:, x]
            prof_left = gray[:, x - delta]
            prof_right= gray[:, x + delta]
            edges     = detect_subpixel_edges(prof, threshold)
            e_left    = detect_subpixel_edges(prof_left, threshold)
            e_right   = detect_subpixel_edges(prof_right, threshold)
            for i in range(0, min(len(edges), len(e_left), len(e_right)) - 1, 2):
                y0, y1   = edges[i], edges[i + 1]
                y0_l, y1_l = e_left[i], e_left[i + 1]
                y0_r, y1_r = e_right[i], e_right[i + 1]
                dy = ((y1_r - y1_l) + (y0_r - y0_l)) / 2.0
                dx = 2.0 * delta
                ang = math.atan2(dy, dx)
                dist_px = abs(y1 - y0) * abs(math.cos(ang))
                all_distances.append(dist_px * mm_per_pixel)
        m = _filter_mean(all_distances)
        if m is not None:
            v_vals.append(m)
            cv2.putText(output, f"{m:.2f} mm", (x_center + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            cv2.line(output, (x_center, 0), (x_center, output.shape[0]),
                     (255, 0, 255), line_thickness)

    # Deviation Table
    rows = len(h_vals) + len(v_vals) + 2
    y_offset = max(30, output.shape[0] - int(rows * spacing))
    cv2.rectangle(output, (5, y_offset - 10),
                  (output.shape[1] - 5, output.shape[0] - 5), table_bg_color, -1)

    table_entries = []
    for i, (d, c) in enumerate(zip(h_vals_def, h_vals)):
        dev = round(c - d, 2)
        color = (0, 255, 0) if abs(dev) <= tolerance else (0, 0, 255)
        text = f"H{i+1}: D:{d:.2f} C:{c:.2f} Δ:{dev:+.2f}"
        table_entries.append((text, color))
    for i, (d, c) in enumerate(zip(v_vals_def, v_vals)):
        dev = round(c - d, 2)
        color = (0, 255, 0) if abs(dev) <= tolerance else (0, 0, 255)
        text = f"V{i+1}: D:{d:.2f} C:{c:.2f} Δ:{dev:+.2f}"
        table_entries.append((text, color))

    for i, (text, color) in enumerate(table_entries):
        y = y_offset + (i + 1) * spacing
        cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness)
    return output

# ==================================================
# UVC acquisition thread (YOLO on motion)
# ==================================================
class UVC_CameraThread(QThread):
    frame_ready = Signal(object)
    fps_updated = Signal(float)
    error_signal = Signal(str)

    def __init__(self, index: int, width=WIDTH, height=HEIGHT):
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self._running = False
        self._cap = None

        # YOLO model (shared across UVC threads by path; lazy-init in run)
        self._yolo = None
        self._class_names = None

    def run(self):
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self.error_signal.emit(f"UVC {self.index}: failed to open")
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self._yolo is None:
            try:
                self._yolo = YOLO(YOLO_MODEL_PATH)
                self._class_names = self._yolo.names
            except Exception as e:
                self.error_signal.emit(f"YOLO load error: {e}")

        self._running = True
        last = time.time()
        frames = 0

        first_frame = True
        background  = None
        update_threshold = 10_00000  # tune for your scene

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.error_signal.emit(f"UVC {self.index}: read failed")
                break

            # Motion gate on a downscaled/blurred copy
            Test_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            gray = cv2.cvtColor(Test_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            if first_frame:
                background = blurred.copy()
                first_frame = False

            diff = cv2.absdiff(background, blurred)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_level = np.sum(thresh)

            obj_present = motion_level >= update_threshold
            background = cv2.addWeighted(background, 0.9, blurred, 0.1, 0)

            if obj_present and self._yolo is not None:
                try:
                    results = self._yolo(frame)
                    dets = results[0]
                    for box in dets.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = self._class_names[class_id] if self._class_names else str(class_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, "Motion Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    cv2.putText(frame, f"YOLO err: {e}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Updating background...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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

# ==================================================
# Basler acquisition thread
# ==================================================
class BAS_CameraThread(QThread):
    frame_ready = Signal(object)
    fps_updated = Signal(float)
    error_signal = Signal(str)

    def __init__(self, index: int, width=WIDTH, height=HEIGHT):
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
                self._camera.Width.SetValue(min(self._camera.Width.Max,  self.width))
                self._camera.Height.SetValue(min(self._camera.Height.Max, self.height))
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

# ==================================================
# Shared camera widget + processor hook
# ==================================================
class _BaseCameraWidget(QWidget):
    def __init__(self, label_text: str):
        super().__init__()
        self.label_text = label_text
        self.zoom_scale = 1.0
        self._latest_packet: Optional[FramePacket] = None
        self._processor: Optional[Callable[[FramePacket], FramePacket]] = None
        self._timer: Optional[QTimer] = None
        self._thread: Optional[QThread] = None
        self._build_ui()
        self._setup_paint_timer()

    def _build_ui(self):
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

        self.setLayout(layout)

    def _setup_paint_timer(self):
        self._timer = QTimer(self)
        interval_ms = int(1000 / max(1, int(TARGET_UI_FPS)))
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._paint_latest)
        self._timer.start()

    def _zoom_in(self):
        self.zoom_scale = min(3.0, self.zoom_scale + 0.1)

    def _zoom_out(self):
        self.zoom_scale = max(0.1, self.zoom_scale - 0.1)

    def _toggle_stream(self, checked: bool):
        if checked:
            self.stream_btn.setText("Stop Stream")
            self._start_thread()
        else:
            self.stream_btn.setText("Start Stream")
            self._stop_thread()

    def _make_thread(self) -> QThread:
        raise NotImplementedError

    def set_processor(self, func: Optional[Callable[[FramePacket], FramePacket]]):
        self._processor = func

    def _start_thread(self):
        if self._thread is not None:
            return
        self._thread = self._make_thread()
        self._thread.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self._thread.fps_updated.connect(self._on_fps, Qt.QueuedConnection)
        self._thread.error_signal.connect(self._on_error, Qt.QueuedConnection)
        self._thread.start()

    def _stop_thread(self):
        if self._thread is not None:
            try:
                self._thread.stop()
            except Exception:
                pass
            self._thread = None

    def _on_frame(self, packet: FramePacket):
        if self._processor is not None:
            try:
                packet = self._processor(packet)
            except Exception as e:
                self.image_label.setText(f"Processor error: {e}")
        self._latest_packet = packet

    def _on_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str):
        self.image_label.setText(msg)
        self.stream_btn.setChecked(False)

    def _paint_latest(self):
        if self._latest_packet is None:
            return
        frame = self._latest_packet.frame
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        sw = max(1, int(w * self.zoom_scale))
        sh = max(1, int(h * self.zoom_scale))
        resized = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0],
                      resized.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def stop_camera_on_close(self):
        self._stop_thread()

# ==================================================
# Concrete widgets
# ==================================================
class UVC_CameraWidget(_BaseCameraWidget):
    def __init__(self, label, index):
        self.index = index
        super().__init__(label)

    def _make_thread(self) -> QThread:
        return UVC_CameraThread(self.index, WIDTH, HEIGHT)

class BAS_CameraWidget(_BaseCameraWidget):
    def __init__(self, label, index, processor: Optional[Callable[[FramePacket], FramePacket]] = None):
        self.index = index
        super().__init__(label)
        if processor:
            self.set_processor(processor)

    def _make_thread(self) -> QThread:
        return BAS_CameraThread(self.index, WIDTH, HEIGHT)

# ==================================================
# Basler measurement processor factory
# ==================================================
def make_basler_processor(template_rgb: Optional[np.ndarray],
                          template_data: Optional[Dict[str, Any]],
                          mm_per_pixel: float = MM_PER_PIXEL_DEFAULT,
                          tolerance: float = TOLERANCE_DEFAULT):
    """
    Returns a function that can be set via set_processor() on BAS_CameraWidget.
    It will align+match the incoming frame to the template and draw measurements.
    If template data is missing, it becomes a no-op pass-through.
    """
    if template_rgb is None or template_data is None:
        def passthrough(packet: FramePacket) -> FramePacket:
            return packet
        return passthrough

    # Keep some state per-processor to reduce matching load
    state = {"every": 10, "count": 0, "last_crop_ok": False}

    def _proc(packet: FramePacket) -> FramePacket:
        frame = packet.frame
        state["count"] += 1
        do_match = (state["count"] % state["every"] == 1) or not state["last_crop_ok"]

        if do_match:
            crop, ok = match_and_crop_aligned(frame, template_rgb)
            state["last_crop_ok"] = ok
            if ok:
                # draw overlay on crop, then paste back for visualization
                out = crop.copy()
                analyze_frame_overlay(out, template_data,
                                      mm_per_pixel=mm_per_pixel,
                                      tolerance=tolerance,
                                      num_parallel=PARALLEL_SAMPLES,
                                      delta=DELTA_OFF,
                                      threshold=EDGE_THRESH,
                                      scale=UI_SCALE)
                # place into top-left for context view (or replace frame fully)
                h, w = out.shape[:2]
                H, W = frame.shape[:2]
                h = min(h, H); w = min(w, W)
                frame[0:h, 0:w] = out[0:h, 0:w]
            # else: leave frame as-is (no match)
        else:
            # keep displaying previous context while saving compute
            pass

        return FramePacket(frame)

    return _proc

# ==================================================
# Main window (same layout)
# ==================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High FPS Multi-Camera Viewer")
        self.setMinimumSize(1400, 800)
        self._init_templates()
        self._init_ui()

    def _init_templates(self):
        self.template_rgb = None
        self.template_data = None
        self.mm_per_pixel = MM_PER_PIXEL_DEFAULT
        self.tolerance = TOLERANCE_DEFAULT

        try:
            if os.path.isfile(TEMPLATE_BMP):
                # Note: matplotlib imread can return float; use cv2 for BGR uint8
                tpl_bgr = cv2.imread(TEMPLATE_BMP, cv2.IMREAD_COLOR)
                self.template_rgb = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[WARN] Failed to load template BMP: {e}")

        try:
            if os.path.isfile(TEMPLATE_JSON):
                import json
                with open(TEMPLATE_JSON, "r") as f:
                    self.template_data = json.load(f)
                # allow overriding mm_per_pixel / tolerance in JSON if present
                self.mm_per_pixel = float(self.template_data.get("mm_per_pixel", self.mm_per_pixel))
                self.tolerance    = float(self.template_data.get("tolerance", self.tolerance))
        except Exception as e:
            print(f"[WARN] Failed to load template JSON: {e}")

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()
        left_grid  = QGridLayout()
        right_grid = QGridLayout()

        self.cameras = []

        # 6 UVC
        for i in range(6):
            cam = UVC_CameraWidget(f"UVC Camera {i}", i)
            self.cameras.append(cam)
            r, c = divmod(i, 2)
            left_grid.addWidget(cam, r, c)

        # 4 Basler with measurement processor
        basler_proc = make_basler_processor(self.template_rgb, self.template_data,
                                           mm_per_pixel=self.mm_per_pixel,
                                           tolerance=self.tolerance)
        for i in range(4):
            cam = BAS_CameraWidget(f"Basler Camera {i}", i, processor=basler_proc)
            self.cameras.append(cam)
            r, c = divmod(i, 2)
            right_grid.addWidget(cam, r, c)

        main_layout.addLayout(left_grid,  2)
        main_layout.addLayout(right_grid, 2)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def closeEvent(self, event):
        for cam in self.cameras:
            cam.stop_camera_on_close()
        super().closeEvent(event)

# ==================================================
# Entrypoint
# ==================================================
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
