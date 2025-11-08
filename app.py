import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import psutil
import os
from ultralytics import YOLO

# PySide6 imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                              QWidget, QHBoxLayout, QPushButton, QSlider, QTextEdit,
                              QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
                              QFormLayout, QFileDialog, QScrollArea, QCheckBox,
                              QComboBox)
from PySide6.QtCore import QTimer, Qt, QSize, Signal
from PySide6.QtGui import QPixmap, QImage, QKeyEvent, QFont, QPalette, QColor

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('yolo_tracker.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- Configuration Management ----------------
class AppConfig:
    """Configuration management class"""
    def __init__(self):
        self.model_path = r"models\best_yolo11_openvino_model"
        self.input_source = 1  # Can be int (camera) or str (file/stream)
        self.confidence = 0.50
        self.iou = 0.50
        self.max_detections = 50
        self.warmup_frames = 5

# Global configuration instance
config = AppConfig()

# ---------------- FPS Meter ----------------
class FpsMeter:
    """
    Minimal-overhead FPS meter:
      - Uses cv2.getTickCount()/getTickFrequency() (fast, high-res)
      - O(1) running-sum ring buffer for short-window average
      - EMA for smooth 'instant' FPS
      - Optional 1s throughput FPS (frames completed / second)
    """
    def __init__(self, window_len=60, ema_alpha=0.1):
        self.freq = cv2.getTickFrequency()
        self.window_len = int(max(1, window_len))
        self.ema_alpha = float(ema_alpha)

        self._buf = [0.0] * self.window_len
        self._idx = 0
        self._count = 0
        self._sum = 0.0
        self._ema = None

        self._tp_last_tick = cv2.getTickCount()
        self._tp_counter = 0
        self._tp_last_value = 0.0
        self._t0 = None

    def start(self):
        self._t0 = cv2.getTickCount()

    def stop(self):
        t1 = cv2.getTickCount()
        dt = (t1 - self._t0) / self.freq  # seconds

        # Ring buffer + running sum (O(1))
        old = self._buf[self._idx]
        self._buf[self._idx] = dt
        self._idx = (self._idx + 1) % self.window_len
        if self._count < self.window_len:
            self._count += 1
            self._sum += dt
        else:
            self._sum += dt - old

        # EMA of dt
        if self._ema is None:
            self._ema = dt
        else:
            a = self.ema_alpha
            self._ema = a * dt + (1 - a) * self._ema

        # Throughput FPS over ~1s window
        self._tp_counter += 1
        elapsed = (t1 - self._tp_last_tick) / self.freq
        if elapsed >= 1.0:
            self._tp_last_value = self._tp_counter / elapsed
            self._tp_counter = 0
            self._tp_last_tick = t1

        return dt * 1000.0  # ms

    @property
    def fps_ema(self):
        if self._ema is None or self._ema == 0:
            return 0.0
        return 1.0 / self._ema

    @property
    def fps_window(self):
        if self._count == 0 or self._sum <= 0:
            return 0.0
        mean_dt = self._sum / self._count
        return 1.0 / mean_dt

    @property
    def fps_throughput(self):
        return float(self._tp_last_value)


class StreamingStats:
    """Online mean/std calculator (Welford) for alert z-scores."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def std(self) -> float:
        var = self.variance()
        return math.sqrt(var) if var > 0 else 0.0

    def z_score(self, value: float) -> Optional[float]:
        if self.count < 2:
            return None
        std = self.std()
        if std <= 1e-6:
            return None
        return (value - self.mean) / std


@dataclass
class MetricReading:
    key: str
    title: str
    value_text: str
    z_text: str
    status: str
    alert_active: bool
    coverage_ready: bool
    alert_message: Optional[str] = None


@dataclass
class BehaviorAlertEvent:
    key: str
    title: str
    message: str
    active: bool


@dataclass
class BehaviorMetricConfig:
    key: str
    title: str
    description: str
    units: str
    window_min: float
    window_max: float
    threshold: float
    direction: str  # "above" or "below"
    min_duration: float
    alert_message: str
    percent_display: bool = False
    skip_when_night: bool = False


@dataclass
class BehaviorMetricState:
    history: Deque[Tuple[float, float]]
    stats: StreamingStats
    breach_since: Optional[float] = None
    alert_active: bool = False


@dataclass
class BehaviorAnalysisResult:
    readings: Dict[str, MetricReading]
    events: List[BehaviorAlertEvent]


class BehaviorAnalyzer:
    """Derives behavior metrics from YOLO results + optical flow."""

    NIGHT_LUMA_THRESHOLD = 28.0

    def __init__(self):
        self.metric_configs = {
            "activity": BehaviorMetricConfig(
                key="activity",
                title="Lethargy / Activity Drop",
                description="Optical-flow slowdown across detections",
                units="px/frame",
                window_min=60.0,
                window_max=120.0,
                threshold=-0.5,
                direction="below",
                min_duration=2.0,
                alert_message="Low DO / post-handling / temperature shock?",
                percent_display=False,
                skip_when_night=True,
            ),
            "crowding": BehaviorMetricConfig(
                key="crowding",
                title="Clustering / Crowding",
                description="Top 10% occupancy grid mass",
                units="%",
                window_min=60.0,
                window_max=120.0,
                threshold=3.0,
                direction="above",
                min_duration=60.0,
                alert_message="Potential water quality issue or disturbance",
                percent_display=True,
            ),
            "edge": BehaviorMetricConfig(
                key="edge",
                title="Edge (Wall-Pacing) Ratio",
                description="Centroids hugging the perimeter band",
                units="%",
                window_min=45.0,
                window_max=90.0,
                threshold=2.5,
                direction="above",
                min_duration=60.0,
                alert_message="Stress / overcrowding / barren tank?",
                percent_display=True,
            ),
            "inflow": BehaviorMetricConfig(
                key="inflow",
                title="Inflow Magnet",
                description="Centroids parked inside inflow ROI",
                units="%",
                window_min=45.0,
                window_max=90.0,
                threshold=3.0,
                direction="above",
                min_duration=60.0,
                alert_message="Possible O₂ seeking / stratification / pump issue",
                percent_display=True,
            ),
        }
        self.metric_states: Dict[str, BehaviorMetricState] = {
            key: BehaviorMetricState(history=deque(), stats=StreamingStats())
            for key in self.metric_configs
        }
        self.prev_gray: Optional[np.ndarray] = None
        self.last_brightness = 0.0
        # ROI polygon defined in normalized coordinates (x, y) for inflow jet/diffuser.
        self.inflow_roi_normalized = np.array(
            [
                (0.72, 0.15),
                (0.95, 0.15),
                (0.95, 0.35),
                (0.72, 0.35),
            ],
            dtype=np.float32,
        )

    def reset(self):
        self.prev_gray = None
        self.last_brightness = 0.0
        for state in self.metric_states.values():
            state.history.clear()
            state.stats.reset()
            state.breach_since = None
            state.alert_active = False

    def analyze(self, frame, boxes) -> Optional[BehaviorAnalysisResult]:
        if frame is None:
            self.prev_gray = None
            return None

        timestamp = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_brightness = float(np.mean(gray))
        box_list = self._extract_boxes(boxes, frame.shape)
        centroids = self._extract_centroids(box_list)

        mean_speed = self._compute_mean_speed(gray, box_list)
        clustering = self._compute_clustering_score(centroids, frame.shape)
        edge_ratio = self._compute_edge_ratio(centroids, frame.shape)
        inflow_ratio = self._compute_inflow_ratio(centroids, frame.shape)

        readings: Dict[str, MetricReading] = {}
        events: List[BehaviorAlertEvent] = []

        metric_inputs = {
            "activity": (mean_speed, self._is_night_condition()),
            "crowding": (clustering, False),
            "edge": (edge_ratio, False),
            "inflow": (inflow_ratio, False),
        }

        for key, (value, skip_for_night) in metric_inputs.items():
            cfg = self.metric_configs[key]
            state = self.metric_states[key]
            skip_alert = cfg.skip_when_night and skip_for_night
            reading, alert_event = self._update_metric(cfg, state, value, timestamp, skip_alert)
            readings[key] = reading
            if alert_event is not None:
                events.append(alert_event)

        return BehaviorAnalysisResult(readings=readings, events=events)

    def _is_night_condition(self) -> bool:
        return self.last_brightness <= self.NIGHT_LUMA_THRESHOLD

    def _extract_boxes(self, boxes, frame_shape) -> List[Tuple[int, int, int, int]]:
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy = boxes.xyxy
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu().numpy()
        else:
            xyxy = np.asarray(xyxy)
        h, w = frame_shape[:2]
        parsed: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in xyxy:
            ix1 = int(max(0, min(w - 1, math.floor(x1))))
            iy1 = int(max(0, min(h - 1, math.floor(y1))))
            ix2 = int(max(0, min(w, math.ceil(x2))))
            iy2 = int(max(0, min(h, math.ceil(y2))))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            parsed.append((ix1, iy1, ix2, iy2))
        return parsed

    def _extract_centroids(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        centers: List[Tuple[int, int]] = []
        for x1, y1, x2, y2 in boxes:
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            centers.append((cx, cy))
        return centers

    def _compute_mean_speed(self, gray: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> Optional[float]:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return None
        if not boxes:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        self.prev_gray = gray
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        total_sum = 0.0
        total_pixels = 0
        for x1, y1, x2, y2 in boxes:
            roi = mag[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            total_sum += float(np.sum(roi))
            total_pixels += roi.size
        if total_pixels == 0:
            return None
        return total_sum / total_pixels

    def _compute_clustering_score(
        self,
        centroids: List[Tuple[int, int]],
        frame_shape: Tuple[int, int, int],
    ) -> Optional[float]:
        if not centroids:
            return None
        grid_size = 20
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        height, width = frame_shape[:2]
        total = float(len(centroids))
        if width <= 0 or height <= 0:
            return None
        for cx, cy in centroids:
            gx = int(np.clip((cx / max(width, 1e-6)) * grid_size, 0, grid_size - 1))
            gy = int(np.clip((cy / max(height, 1e-6)) * grid_size, 0, grid_size - 1))
            grid[gy, gx] += 1.0
        grid /= total
        flattened = np.sort(grid.ravel())
        top_cells = max(1, int(flattened.size * 0.10))
        score = float(np.sum(flattened[-top_cells:]))
        return score

    def _compute_edge_ratio(
        self,
        centroids: List[Tuple[int, int]],
        frame_shape: Tuple[int, int, int],
    ) -> Optional[float]:
        if not centroids:
            return None
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return None
        margin_x = width * 0.12
        margin_y = height * 0.12
        edge_hits = 0
        for cx, cy in centroids:
            if cx <= margin_x or cx >= (width - margin_x) or cy <= margin_y or cy >= (height - margin_y):
                edge_hits += 1
        return edge_hits / len(centroids)

    def _compute_inflow_ratio(
        self,
        centroids: List[Tuple[int, int]],
        frame_shape: Tuple[int, int, int],
    ) -> Optional[float]:
        if not centroids:
            return None
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return None
        polygon = np.array(
            [
                (
                    int(np.clip(x * width, 0, width - 1)),
                    int(np.clip(y * height, 0, height - 1)),
                )
                for x, y in self.inflow_roi_normalized
            ],
            dtype=np.int32,
        )
        hits = 0
        for cx, cy in centroids:
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
                hits += 1
        return hits / len(centroids)

    def _update_metric(
        self,
        cfg: BehaviorMetricConfig,
        state: BehaviorMetricState,
        value: Optional[float],
        timestamp: float,
        skip_alert: bool,
    ) -> Tuple[MetricReading, Optional[BehaviorAlertEvent]]:
        alert_event: Optional[BehaviorAlertEvent] = None
        status = "No detections"
        z_text = "z n/a"
        value_text = "--"
        coverage_ready = False
        z_score = None

        if value is None:
            state.history.clear()
            if state.alert_active:
                state.alert_active = False
                state.breach_since = None
                alert_event = BehaviorAlertEvent(
                    key=cfg.key,
                    title=cfg.title,
                    message="Condition cleared (no detections)",
                    active=False,
                )
            reading = MetricReading(
                key=cfg.key,
                title=cfg.title,
                value_text=value_text,
                z_text=z_text,
                status=status,
                alert_active=False,
                coverage_ready=False,
            )
            return reading, alert_event

        state.history.append((timestamp, value))
        while state.history and (timestamp - state.history[0][0]) > cfg.window_max:
            state.history.popleft()

        coverage = timestamp - state.history[0][0] if len(state.history) > 1 else 0.0
        coverage_ready = coverage >= cfg.window_min
        aggregated = float(np.mean([sample for _, sample in state.history])) if state.history else value

        if cfg.percent_display:
            value_text = f"{aggregated * 100:.1f} {cfg.units}"
        else:
            value_text = f"{aggregated:.3f} {cfg.units}"

        if coverage_ready:
            if not (skip_alert and cfg.skip_when_night):
                state.stats.update(aggregated)
            z_score = state.stats.z_score(aggregated)
        else:
            status = f"Baseline building ({coverage:.0f}/{cfg.window_min:.0f}s)"

        if z_score is not None:
            z_text = f"z {z_score:+.2f}"

        meets_threshold = False
        if coverage_ready and z_score is not None and not skip_alert:
            if cfg.direction == "below":
                meets_threshold = z_score <= cfg.threshold
            else:
                meets_threshold = z_score >= cfg.threshold

        if skip_alert and coverage_ready:
            status = "Night mode (muted)"
        elif coverage_ready and not state.alert_active:
            status = "Tracking (z pending)" if z_score is None else "Stable"

        if meets_threshold:
            if state.breach_since is None:
                state.breach_since = timestamp
            elapsed = timestamp - state.breach_since
            if elapsed >= cfg.min_duration and not state.alert_active:
                state.alert_active = True
                alert_event = BehaviorAlertEvent(
                    key=cfg.key,
                    title=cfg.title,
                    message=cfg.alert_message,
                    active=True,
                )
        else:
            if state.alert_active:
                state.alert_active = False
                alert_event = BehaviorAlertEvent(
                    key=cfg.key,
                    title=cfg.title,
                    message="Condition cleared",
                    active=False,
                )
            state.breach_since = None

        if state.alert_active:
            status = cfg.alert_message

        reading = MetricReading(
            key=cfg.key,
            title=cfg.title,
            value_text=value_text,
            z_text=z_text,
            status=status,
            alert_active=state.alert_active,
            coverage_ready=coverage_ready,
            alert_message=cfg.alert_message if state.alert_active else None,
        )
        return reading, alert_event


class BehaviorAnalysisThread:
    """Background worker that runs BehaviorAnalyzer off the UI thread."""

    def __init__(self):
        self.analyzer = BehaviorAnalyzer()
        self.metric_configs = self.analyzer.metric_configs
        self._task_queue: "queue.Queue[Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]" = queue.Queue(maxsize=1)
        self._result_queue: "queue.Queue[Optional[BehaviorAnalysisResult]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, name="BehaviorAnalysisThread", daemon=True)
        self._thread.start()

    def submit(self, frame: Optional[np.ndarray], boxes) -> None:
        """Queue the latest frame + boxes snapshot for analysis."""
        if frame is None or self._stop_event.is_set():
            return
        snapshot = (frame.copy(), self._snapshot_boxes(boxes))
        try:
            self._task_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                pass
            self._task_queue.put_nowait(snapshot)

    def consume_results(self) -> List[BehaviorAnalysisResult]:
        """Retrieve all available analysis outputs."""
        results: List[BehaviorAnalysisResult] = []
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                results.append(item)
        return results

    def reset(self):
        """Reset analyzer state and clear pending work/results."""
        self.analyzer.reset()
        self._clear_queue(self._task_queue)
        self._clear_queue(self._result_queue)

    def shutdown(self):
        """Signal the worker to exit."""
        self._stop_event.set()
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                frame, boxes_snapshot = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            boxes_obj = None if boxes_snapshot is None else SimpleNamespace(xyxy=boxes_snapshot)
            result = self.analyzer.analyze(frame, boxes_obj)
            if result is not None:
                self._result_queue.put(result)

    @staticmethod
    def _snapshot_boxes(boxes) -> Optional[np.ndarray]:
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return None
        xyxy = boxes.xyxy
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu().numpy()
        else:
            xyxy = np.asarray(xyxy)
        return xyxy.copy()

    @staticmethod
    def _clear_queue(q: "queue.Queue"):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return


# ---------------- Source helpers ----------------
def validate_source(src: str):
    """
    Basic sanity check and log:
      - Accepts camera index (e.g. "0")
      - Accepts RTSP/HTTP(S) URLs
      - Accepts existing file paths
    """
    src_str = str(src)
    is_cam_index = src_str.isdigit()
    is_stream = src_str.startswith(("rtsp://", "http://", "https://"))

    if not is_cam_index and not is_stream and not Path(src_str).exists():
        logger.error(f"Video/stream not found or invalid: {src_str}")
        raise FileNotFoundError(f"Video/stream not found or invalid: {src_str}")

    if is_cam_index:
        logger.info(f"Using camera index {src_str} as source")
    elif is_stream:
        logger.info(f"Using network stream source: {src_str}")
    else:
        logger.info(f"Using file source: {src_str}")


def resolve_source(src: str):
    """Return int for camera index strings, else return string unchanged."""
    src_str = str(src)
    if src_str.isdigit():
        return int(src_str)
    return src_str


# ---------------- Configuration Panel ----------------
class ConfigurationPanel(QWidget):
    """Configuration panel widget for YOLO parameters"""
    config_changed = Signal()  # Signal emitted when config changes
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        """Setup the configuration panel UI"""
        # Main scroll area for better usability
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        self.scroll_layout = layout
        
        # Model Configuration Group
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout(model_group)
        self.model_group = model_group
        
        # Model Path
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select YOLO model file (.pt, .onnx, .engine)")
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model_path)
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.browse_model_btn)
        model_layout.addRow("Model Path:", model_path_layout)
        
        # Source Configuration Group
        source_group = QGroupBox("Source Configuration")
        source_layout = QFormLayout(source_group)
        
        # Input Source
        self.input_source_edit = QLineEdit()
        self.input_source_edit.setPlaceholderText("Camera index (e.g., 0) or file path or RTSP URL")
        source_layout.addRow("Input Source:", self.input_source_edit)
        
        # Detection Parameters Group
        detection_group = QGroupBox("Detection Parameters")
        detection_layout = QFormLayout(detection_group)
        
        # Confidence threshold
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setDecimals(2)
        detection_layout.addRow("Confidence:", self.confidence_spin)
        
        # IOU threshold
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        detection_layout.addRow("IoU Threshold:", self.iou_spin)
        
        # Max detections
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)
        detection_layout.addRow("Max Detections:", self.max_det_spin)
        
        # Warmup frames
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 100)
        detection_layout.addRow("Warmup Frames:", self.warmup_spin)
        
        # Control Buttons Group
        control_group = QGroupBox("Detection Control")
        control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        control_layout = QVBoxLayout(control_group)
        
        # Start/Stop button
        self.start_button = QPushButton("Start Detection")
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        control_layout.addWidget(self.start_button)
        
        # Status displays
        status_layout = QVBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.detection_label = QLabel("Detected: 0")
        self.fps_label.setStyleSheet("color: white;")
        self.detection_label.setStyleSheet("color: white;")
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.detection_label)
        control_layout.addLayout(status_layout)
        
        # Apply changes button
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.apply_button.clicked.connect(self.apply_configuration)
        control_layout.addWidget(self.apply_button)
        
        # Style all group boxes
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        source_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        detection_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Add all groups to main layout
        self.model_group.hide()  # keep configuration logic but hide UI as requested
        layout.addWidget(model_group)
        layout.addWidget(source_group)
        layout.addWidget(detection_group)
        layout.addWidget(control_group)
        self.behavior_insert_index = layout.count()
        layout.addStretch()
        
        # Set up scroll area
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #888;
            }
        """)
        
        # Main layout with dark background
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 10, 10, 10)  # Left margin smaller for spacing
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
                color: white;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #4CAF50;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                color: white;
            }
        """)
        main_layout.addWidget(scroll_area)
        
        # Connect signals for config changes
        self.connect_config_signals()

    def add_behavior_widget(self, widget: QWidget):
        """Insert external widgets just above the stretch (after control group)."""
        if not hasattr(self, "scroll_layout") or self.scroll_layout is None:
            return
        insert_index = getattr(self, "behavior_insert_index", self.scroll_layout.count())
        self.scroll_layout.insertWidget(insert_index, widget)
        self.behavior_insert_index = insert_index + 1
        
    def connect_config_signals(self):
        """Connect UI signals to config update methods"""
        widgets = [
            self.model_path_edit, self.input_source_edit,
            self.confidence_spin, self.iou_spin,
            self.max_det_spin, self.warmup_spin
        ]
        for widget in widgets:
            if hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.on_config_changed)
            elif hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.on_config_changed)
                
    def on_config_changed(self):
        """Handle configuration changes"""
        self.config_changed.emit()
        
    def load_config(self):
        """Load current configuration into UI"""
        self.model_path_edit.setText(self.config.model_path)
        self.input_source_edit.setText(str(self.config.input_source))
        self.confidence_spin.setValue(self.config.confidence)
        self.iou_spin.setValue(self.config.iou)
        self.max_det_spin.setValue(self.config.max_detections)
        self.warmup_spin.setValue(self.config.warmup_frames)
        
    def apply_configuration(self):
        """Apply configuration from UI to config object"""
        try:
            # Update config with UI values
            self.config.model_path = self.model_path_edit.text().strip()
            
            # Handle input source (could be int or string)
            input_source_text = self.input_source_edit.text().strip()
            if input_source_text.isdigit():
                self.config.input_source = int(input_source_text)
            else:
                self.config.input_source = input_source_text
                
            self.config.confidence = float(self.confidence_spin.value())
            self.config.iou = float(self.iou_spin.value())
            self.config.max_detections = int(self.max_det_spin.value())
            self.config.warmup_frames = int(self.warmup_spin.value())
            
            self.config_changed.emit()
            
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            
    def browse_model_path(self):
        """Open file dialog to select model path"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "Model Files (*.pt *.onnx *.engine *.torchscript);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def update_status(self, fps, detection_count):
        """Update status displays"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.detection_label.setText(f"Detected: {detection_count}")
        
    def set_start_button_state(self, is_running):
        """Update start button appearance based on detection state"""
        if is_running:
            self.start_button.setText("Stop Detection")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #c62828;
                }
            """)
        else:
            self.start_button.setText("Start Detection")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)

# ---------------- PySide6 Main Window ----------------
class YoloTrackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Sturgeon Monitor (PySide6)")
        self.setMinimumSize(1200, 800)
        
        # Initialize YOLO model and processing variables
        self.model = None
        self.results_gen = None
        self.meter = FpsMeter(window_len=60, ema_alpha=0.1)
        self.frames_done = 0
        self.is_processing = False
        self.init_heatmap_state()
        self.behavior_worker = BehaviorAnalysisThread()
        self.behavior_metric_configs = self.behavior_worker.metric_configs
        self.behavior_cards: Dict[str, Dict[str, QLabel]] = {}
        self.behavior_enabled = True
        self.active_alerts: Dict[str, str] = {}
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for frame processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        
    def setup_ui(self):
        """Setup the user interface with horizontal layout"""
        # Set overall dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side: Video feed
        self.setup_video_panel(main_layout)
        
        # Right side: Configuration panel
        self.setup_config_panel(main_layout)
        
    def setup_video_panel(self, parent_layout):
        """Setup the video display panel on the left"""
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(10, 10, 5, 10)  # Right margin smaller for spacing
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                background-color: #222;
                border-radius: 4px;
                color: white;
            }
        """)
        video_layout.addWidget(self.video_label)
        
        # Status log with dark theme
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        video_layout.addWidget(self.log_text)
        self.setup_heatmap_section(video_layout)
        
        # Add to main layout with stretch factor
        parent_layout.addWidget(video_container, 2)  # Takes 2/3 of space

    def create_behavior_widget(self):
        """Build the behavior analytics card group."""
        behavior_group = QGroupBox("Behavior Analysis")
        behavior_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 18px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
        layout = QVBoxLayout(behavior_group)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        self.behavior_toggle = QCheckBox("Enable Behavior Analysis")
        self.behavior_toggle.setChecked(True)
        self.behavior_toggle.toggled.connect(self.on_behavior_enabled_toggled)
        layout.addWidget(self.behavior_toggle)

        # Ensure consistent ordering (Lethargy first, others below)
        ordered_keys = ["activity", "crowding", "edge", "inflow"]
        for key in ordered_keys:
            cfg = self.behavior_metric_configs.get(key)
            if cfg is None:
                continue
            card = self._create_behavior_card(cfg.title, cfg.description)
            layout.addWidget(card["widget"])
            self.behavior_cards[key] = card
        return behavior_group

    def create_alert_widget(self):
        alert_group = QGroupBox("Behavior Alerts")
        alert_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 16px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
        layout = QVBoxLayout(alert_group)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(4)

        self.alert_label = QLabel("No active alerts")
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet("color: #f5f5f5; font-size: 12px;")
        layout.addWidget(self.alert_label)
        return alert_group

    def _create_behavior_card(self, title: str, description: str):
        card_widget = QWidget()
        card_layout = QVBoxLayout(card_widget)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #bbbbbb; font-size: 11px;")

        value_label = QLabel("--")
        value_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #80cbc4;")

        status_label = QLabel("Initializing...")
        status_label.setStyleSheet("color: #c8c8c8; font-size: 11px;")

        card_layout.addWidget(title_label)
        card_layout.addWidget(desc_label)
        card_layout.addWidget(value_label)
        card_layout.addWidget(status_label)

        return {
            "widget": card_widget,
            "value_label": value_label,
            "status_label": status_label,
        }

    def update_behavior_ui(self, readings: Dict[str, MetricReading]):
        if not self.behavior_enabled or not readings:
            return

        for key, reading in readings.items():
            card = self.behavior_cards.get(key)
            if not card:
                continue

            value_label: QLabel = card["value_label"]  # type: ignore[index]
            status_label: QLabel = card["status_label"]  # type: ignore[index]

            value_label.setText(f"{reading.value_text} | {reading.z_text}")
            status_label.setText(reading.status)

            if reading.alert_active:
                status_label.setStyleSheet("color: #ff8a80; font-weight: bold; font-size: 11px;")
                value_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffab91;")
            elif reading.coverage_ready:
                status_label.setStyleSheet("color: #8bc34a; font-size: 11px;")
                value_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #a5d6a7;")
            else:
                status_label.setStyleSheet("color: #c8c8c8; font-size: 11px;")
                value_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #80cbc4;")

    def set_behavior_cards_disabled(self, disabled: bool):
        for card in self.behavior_cards.values():
            value_label: QLabel = card["value_label"]  # type: ignore[index]
            status_label: QLabel = card["status_label"]  # type: ignore[index]
            if disabled:
                value_label.setText("-- | z n/a")
                value_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #888888;")
                status_label.setText("Behavior analysis disabled")
                status_label.setStyleSheet("color: #888888; font-size: 11px;")
            else:
                status_label.setText("Waiting for data...")
                status_label.setStyleSheet("color: #c8c8c8; font-size: 11px;")
                value_label.setText("-- | z n/a")
                value_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #80cbc4;")

    def refresh_alert_display(self):
        if not hasattr(self, "alert_label"):
            return
        if not self.behavior_enabled:
            self.alert_label.setText("Behavior analysis disabled")
            self.alert_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
            return
        if not self.active_alerts:
            self.alert_label.setText("No active alerts")
            self.alert_label.setStyleSheet("color: #8bc34a; font-size: 12px;")
            return
        lines = []
        for key, message in self.active_alerts.items():
            cfg = self.behavior_metric_configs.get(key)
            if cfg is None:
                lines.append(f"- {message}")
            else:
                lines.append(f"- {cfg.title}: {message}")
        self.alert_label.setText("\n".join(lines))
        self.alert_label.setStyleSheet("color: #ffab91; font-size: 12px;")

    def drain_behavior_results(self):
        if (not hasattr(self, "behavior_worker") or self.behavior_worker is None
                or not self.behavior_enabled):
            return
        analyses = self.behavior_worker.consume_results()
        for analysis in analyses:
            if analysis is None:
                continue
            self.update_behavior_ui(analysis.readings)
            for event in analysis.events:
                prefix = "ALERT" if event.active else "RESOLVED"
                self.log_message(f"[{prefix}] {event.title}: {event.message}")
                if event.active:
                    logger.warning(f"{prefix} {event.title}: {event.message}")
                    self.active_alerts[event.key] = event.message
                else:
                    logger.info(f"{prefix} {event.title}: {event.message}")
                    self.active_alerts.pop(event.key, None)
        self.refresh_alert_display()

    def setup_config_panel(self, parent_layout):
        """Setup the configuration panel on the right"""
        self.config_panel = ConfigurationPanel(config)
        self.config_panel.start_button.clicked.connect(self.toggle_detection)
        self.config_panel.config_changed.connect(self.on_config_changed)
        behavior_widget = self.create_behavior_widget()
        self.config_panel.add_behavior_widget(behavior_widget)
        alert_widget = self.create_alert_widget()
        self.config_panel.add_behavior_widget(alert_widget)
        self.refresh_alert_display()
        parent_layout.addWidget(self.config_panel, 1)  # Takes 1/3 of space

    def on_behavior_enabled_toggled(self, checked: bool):
        self.behavior_enabled = checked
        self.behavior_worker.reset()
        self.active_alerts.clear()
        if not checked:
            self.set_behavior_cards_disabled(True)
        else:
            self.set_behavior_cards_disabled(False)
        self.refresh_alert_display()

    def init_heatmap_state(self):
        """Initialize heatmap buffers and defaults"""
        self.heatmap_width = 480
        self.heatmap_height = 240
        self.heatmap_accum = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)
        self.heatmap_enabled = True
        self.heatmap_decay = 0.92
        self.heatmap_intensity = 1.5
        self.heatmap_radius = 12
        self.colormap_options = self._build_colormap_options()
        default_colormap = self.colormap_options[0][1] if self.colormap_options else cv2.COLORMAP_JET
        self.heatmap_colormap = default_colormap
        self.heatmap_label = None

    def _build_colormap_options(self):
        preferred = ["INFERNO", "PLASMA", "MAGMA", "TURBO", "JET", "HOT"]
        options = []
        for name in preferred:
            attr = f"COLORMAP_{name}"
            if hasattr(cv2, attr):
                pretty_name = name.capitalize()
                options.append((pretty_name, getattr(cv2, attr)))
        if not options:
            options.append(("Jet", cv2.COLORMAP_JET))
        return options

    def setup_heatmap_section(self, parent_layout):
        """Create the heatmap display with adjacent settings"""
        heatmap_container = QWidget()
        heatmap_layout = QHBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(0, 12, 0, 0)
        heatmap_layout.setSpacing(12)

        self.heatmap_label = QLabel("Heatmap initializing...")
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setMinimumSize(360, 220)
        self.heatmap_label.setStyleSheet("""
            QLabel {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #1b1b1b;
                color: #bbbbbb;
            }
        """)
        heatmap_layout.addWidget(self.heatmap_label, 3)

        settings_group = QGroupBox("Heatmap Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.heatmap_enable_checkbox = QCheckBox("Enable Heatmap")
        self.heatmap_enable_checkbox.setChecked(self.heatmap_enabled)
        self.heatmap_enable_checkbox.toggled.connect(self.on_heatmap_enabled_toggled)
        settings_layout.addRow(self.heatmap_enable_checkbox)

        self.heatmap_decay_spin = QDoubleSpinBox()
        self.heatmap_decay_spin.setRange(0.50, 0.999)
        self.heatmap_decay_spin.setDecimals(3)
        self.heatmap_decay_spin.setSingleStep(0.01)
        self.heatmap_decay_spin.setValue(self.heatmap_decay)
        self.heatmap_decay_spin.valueChanged.connect(self.on_heatmap_decay_changed)
        settings_layout.addRow("Decay:", self.heatmap_decay_spin)

        self.heatmap_intensity_spin = QDoubleSpinBox()
        self.heatmap_intensity_spin.setRange(0.1, 10.0)
        self.heatmap_intensity_spin.setSingleStep(0.1)
        self.heatmap_intensity_spin.setValue(self.heatmap_intensity)
        self.heatmap_intensity_spin.valueChanged.connect(self.on_heatmap_intensity_changed)
        settings_layout.addRow("Intensity:", self.heatmap_intensity_spin)

        self.heatmap_radius_spin = QSpinBox()
        self.heatmap_radius_spin.setRange(1, 50)
        self.heatmap_radius_spin.setValue(self.heatmap_radius)
        self.heatmap_radius_spin.valueChanged.connect(self.on_heatmap_radius_changed)
        settings_layout.addRow("Radius:", self.heatmap_radius_spin)

        self.heatmap_colormap_combo = QComboBox()
        for name, _ in self.colormap_options:
            self.heatmap_colormap_combo.addItem(name)
        self.heatmap_colormap_combo.setCurrentText(self._colormap_name_for_code(self.heatmap_colormap))
        self.heatmap_colormap_combo.currentTextChanged.connect(self.on_heatmap_colormap_changed)
        settings_layout.addRow("Colormap:", self.heatmap_colormap_combo)

        self.reset_heatmap_button = QPushButton("Reset Heatmap")
        self.reset_heatmap_button.clicked.connect(self.reset_heatmap)
        settings_layout.addRow(self.reset_heatmap_button)

        heatmap_layout.addWidget(settings_group, 2)
        parent_layout.addWidget(heatmap_container)
        self.render_heatmap()

    def _colormap_name_for_code(self, code):
        for name, value in self.colormap_options:
            if value == code:
                return name
        return self.colormap_options[0][0]

    def on_heatmap_enabled_toggled(self, checked):
        self.heatmap_enabled = checked
        if not checked and self.heatmap_label is not None:
            self.heatmap_label.setPixmap(QPixmap())
            self.heatmap_label.setText("Heatmap disabled")
        else:
            self.render_heatmap()

    def on_heatmap_decay_changed(self, value):
        self.heatmap_decay = float(value)

    def on_heatmap_intensity_changed(self, value):
        self.heatmap_intensity = float(value)

    def on_heatmap_radius_changed(self, value):
        self.heatmap_radius = int(value)

    def on_heatmap_colormap_changed(self, name):
        for option_name, option_value in self.colormap_options:
            if option_name == name:
                self.heatmap_colormap = option_value
                break
        self.render_heatmap()

    def reset_heatmap(self):
        self.heatmap_accum = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)
        if self.heatmap_label is not None:
            self.render_heatmap()

    def update_heatmap(self, res):
        if not self.heatmap_enabled or self.heatmap_accum is None:
            return

        self.heatmap_accum *= self.heatmap_decay

        boxes = getattr(res, "boxes", None)
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy
            if hasattr(xyxy, "detach"):
                xyxy = xyxy.detach().cpu().numpy()
            else:
                xyxy = np.asarray(xyxy)

            orig_shape = getattr(res, "orig_shape", None)
            if orig_shape is None and hasattr(boxes, "orig_shape"):
                orig_shape = boxes.orig_shape
            if orig_shape is None:
                orig_h, orig_w = self.heatmap_height, self.heatmap_width
            else:
                orig_h, orig_w = orig_shape[:2]

            scale_x = self.heatmap_width / max(orig_w, 1)
            scale_y = self.heatmap_height / max(orig_h, 1)

            for x1, y1, x2, y2 in xyxy:
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                u = int(np.clip(cx * scale_x, 0, self.heatmap_width - 1))
                v = int(np.clip(cy * scale_y, 0, self.heatmap_height - 1))
                cv2.circle(self.heatmap_accum, (u, v), self.heatmap_radius, self.heatmap_intensity, -1)

        self.render_heatmap()

    def render_heatmap(self):
        if self.heatmap_label is None:
            return

        if not self.heatmap_enabled:
            self.heatmap_label.setPixmap(QPixmap())
            self.heatmap_label.setText("Heatmap disabled")
            return

        if self.heatmap_accum is None:
            self.reset_heatmap()
            return

        heatmap_data = self.heatmap_accum.copy()
        if heatmap_data.size == 0:
            return

        if np.max(heatmap_data) > 0:
            normalized = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = heatmap_data
        normalized = normalized.astype(np.uint8)
        colored = cv2.applyColorMap(normalized, self.heatmap_colormap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        h, w, ch = colored.shape
        q_img = QImage(colored.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        target_size = self.heatmap_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = QSize(w, h)
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.heatmap_label.setPixmap(scaled)
        self.heatmap_label.setText("")
        
    def log_message(self, message):
        """Add message to log widget"""
        self.log_text.append(f"{message}")
        # Keep only last 100 lines
        lines = self.log_text.toPlainText().split('\n')
        if len(lines) > 100:
            self.log_text.setPlainText('\n'.join(lines[-100:]))
            
    def on_config_changed(self):
        """Handle configuration changes"""
        if self.is_processing:
            self.log_message("Configuration changed. Stop detection to apply changes.")
            
    def toggle_detection(self):
        """Toggle detection on/off"""
        if not self.is_processing:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        """Start the detection process"""
        if self.is_processing:
            return
            
        try:
            # Apply current configuration
            self.config_panel.apply_configuration()
            
            # Process priority (best-effort)
            try:
                p = psutil.Process(os.getpid())
                if os.name == 'nt':
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                    logger.info("Process priority set to HIGH_PRIORITY_CLASS (Windows)")
                else:
                    p.nice(-20)
                    logger.info("Process priority set to -20 (Unix-like systems)")
            except Exception as e:
                logger.warning(f"Failed to set process priority: {e}")

            # Load model
            if not Path(config.model_path).exists():
                logger.error(f"Model not found: {config.model_path}")
                raise FileNotFoundError(f"Model not found: {config.model_path}")

            logger.info("Loading YOLO model for segmentation task...")
            self.model = YOLO(config.model_path, task="detect")  # use task="detect" if detector
            logger.info("Model loaded successfully")

            # Warmup (dummy frames) - makes measured FPS more stable/realistic
            if config.warmup_frames > 0:
                logger.info("Warming up model with dummy frames...")
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                dummy_batch = [dummy_frame] * config.warmup_frames
                try:
                    _ = self.model.predict(
                        source=dummy_batch,
                        conf=config.confidence,
                        iou=config.iou,
                        max_det=config.max_detections,
                        verbose=False
                    )
                    logger.info("Model warmup completed")
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")

            # Validate and resolve source for Ultralytics stream loader
            validate_source(config.input_source)
            source_obj = resolve_source(config.input_source)

            logger.info("Creating Ultralytics streaming generator...")
            self.results_gen = self.model.track(
                source=source_obj,
                stream=True,
                conf=config.confidence,
                iou=config.iou,
                max_det=config.max_detections,
                verbose=False
            )
            
            # Reset counters
            self.meter = FpsMeter(window_len=60, ema_alpha=0.1)
            self.frames_done = 0
            self.reset_heatmap()
            if self.behavior_enabled:
                self.behavior_worker.reset()
                self.active_alerts.clear()
                self.refresh_alert_display()
            
            # Start processing
            self.is_processing = True
            self.config_panel.set_start_button_state(True)
            self.timer.start(1)  # Process as fast as possible
            
            self.log_message("Detection started successfully")
            logger.info("Starting processing loop (Ultralytics stream=True)")
            
        except Exception as e:
            error_msg = f"Failed to start detection: {e}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg)
            
    def stop_detection(self):
        """Stop the detection process"""
        if not self.is_processing:
            return
            
        self.is_processing = False
        self.timer.stop()
        self.drain_behavior_results()
        self.config_panel.set_start_button_state(False)
        
        if self.frames_done > 0:
            self.log_message(f"Detection stopped. Total frames processed: {self.frames_done}")
            logger.info(f"User stopped detection. Frames processed: {self.frames_done}")
        else:
            self.log_message("Detection stopped")
            
    def process_frame(self):
        """Process a single frame (called by timer)"""
        if not self.is_processing or self.results_gen is None:
            return

        self.drain_behavior_results()
            
        try:
            self.meter.start()  # measure end-to-end: next() + plot + display

            try:
                res = next(self.results_gen)  # Ultralytics handles capture + inference
            except StopIteration:
                self.log_message(f"End of stream. Total frames: {self.frames_done}")
                logger.info(f"End of stream. Total frames: {self.frames_done}")
                self.stop_detection()
                return
            except Exception as e:
                error_msg = f"Error getting next result from generator: {e}"
                self.log_message(error_msg)
                logger.error(error_msg, exc_info=True)
                self.stop_detection()
                return

            # Annotated frame (BGR)
            self.update_heatmap(res)
            annotated = res.plot()
            orig_frame = getattr(res, "orig_img", None)
            if orig_frame is None:
                orig_frame = annotated

            # Stop timing for full loop
            loop_time_ms = self.meter.stop()
            self.frames_done += 1

            # Update FPS displays
            fps_display = self.meter.fps_ema
            throughput = self.meter.fps_throughput
            
            # Update both main window and config panel
            self.config_panel.update_status(fps_display, len(res.boxes) if res.boxes is not None else 0)
            
            # Add loop time and throughput to log every 100 frames
            if self.frames_done % 100 == 0:
                debug_msg = (f"Frame {self.frames_done} | FPS(EMA) {fps_display:.2f} | "
                           f"{loop_time_ms:.1f} ms | TP {throughput:.2f}")
                self.log_message(debug_msg)
                logger.debug(debug_msg)

            if self.behavior_enabled:
                self.behavior_worker.submit(orig_frame, res.boxes)
            
            # Convert BGR to RGB for Qt
            rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # Create QImage and convert to QPixmap
            q_img = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Display the image
            self.video_label.setPixmap(scaled_pixmap)

        except Exception as e:
            error_msg = f"Unexpected error in frame processing: {e}"
            self.log_message(error_msg)
            logger.error(error_msg, exc_info=True)
            self.stop_detection()
            
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Escape or event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_detection()
        else:
            super().keyPressEvent(event)
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_detection()
        if hasattr(self, "behavior_worker") and self.behavior_worker is not None:
            self.behavior_worker.shutdown()
        logger.info("Resources cleaned up. Application terminated.")
        event.accept()


# ---------------- Main ----------------
def main():
    logger.info("Starting YOLO PySide6 application")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = YoloTrackerWindow()
    window.show()
    
    logger.info("PySide6 application started successfully")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
