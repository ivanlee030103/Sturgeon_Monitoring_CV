import logging
import sys
from pathlib import Path
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
        
    def setup_config_panel(self, parent_layout):
        """Setup the configuration panel on the right"""
        self.config_panel = ConfigurationPanel(config)
        self.config_panel.start_button.clicked.connect(self.toggle_detection)
        self.config_panel.config_changed.connect(self.on_config_changed)
        parent_layout.addWidget(self.config_panel, 1)  # Takes 1/3 of space

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
