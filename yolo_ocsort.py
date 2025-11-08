"""Fish movement tracking application using YOLOv11 and Gradio."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]

from behavior_lite import BehaviorLite


MODEL_PATH = Path(__file__).with_name("best_yolo11.pt").resolve()
BYTETRACK_CONFIG_PATH = Path(__file__).with_name("bytetrack_lowfps.yaml").resolve()
DEFAULT_CAMERA_INDEX = 0
MAX_CAMERA_INDEX = 8
DEFAULT_TRAIL_LENGTH = 32
MIN_TRAIL_LENGTH = 4
MAX_TRAIL_LENGTH = 128
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TRACK_COLOR = (0, 255, 255)
FPS_SMOOTHING = 0.9
DEFAULT_CONF_THRESHOLD = 0.5
MIN_CONF_THRESHOLD = 0.1
MAX_CONF_THRESHOLD = 1.0
DEFAULT_IOU_THRESHOLD = 0.5
MIN_IOU_THRESHOLD = 0.1
MAX_IOU_THRESHOLD = 0.9
BYTETRACK_CONFIG = {
	"tracker_type": "bytetrack",
	"track_high_thresh": 0.6,
	"track_low_thresh": 0.2,
	"new_track_thresh": 0.35,
	"track_buffer": 30,
	"match_thresh": 0.7,
	"conf_thres": 0.35,
	"mot20": False,
	"fuse_score": True,
}
DEFAULT_HEATMAP_DECAY = 0.85
MIN_HEATMAP_DECAY = 0.5
MAX_HEATMAP_DECAY = 0.99
DEFAULT_HEATMAP_GAIN = 0.8
MIN_HEATMAP_GAIN = 0.1
MAX_HEATMAP_GAIN = 3.0
DEFAULT_HEATMAP_RADIUS = 35
MIN_HEATMAP_RADIUS = 5
MAX_HEATMAP_RADIUS = 120
DEFAULT_HEATMAP_BLUR = 15
MIN_HEATMAP_BLUR = 0
MAX_HEATMAP_BLUR = 51
DEFAULT_METERS_PER_PIXEL = 0.01


class FishTracker:
	"""Wrapper around a YOLOv11 model with simple trajectory drawing."""

	def __init__(self, model_path: Path) -> None:
		self._model_path = model_path
		self._model: YOLO | None = None
		self._model_lock = threading.Lock()
		self._trail_length = DEFAULT_TRAIL_LENGTH
		self._track_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
			self._new_history
		)
		self._heatmap: np.ndarray | None = None
		self._heatmap_decay = DEFAULT_HEATMAP_DECAY
		self._heatmap_gain = DEFAULT_HEATMAP_GAIN
		self._heatmap_radius = DEFAULT_HEATMAP_RADIUS
		self._heatmap_blur = DEFAULT_HEATMAP_BLUR
		self._ensure_bytetrack_config()
		self._behavior = BehaviorLite(meters_per_pixel=DEFAULT_METERS_PER_PIXEL)
		self._behavior_context = "pre_feed"
		self._latest_behavior_metrics: Optional[Dict[str, float]] = None
		self._latest_behavior_alerts: List[str] = []
		self._pending_behavior_event: Optional[Tuple[Dict[str, float], List[str]]] = None

	def _new_history(self) -> Deque[Tuple[int, int]]:
		return deque(maxlen=self._trail_length)

	def _ensure_bytetrack_config(self) -> None:
		lines: List[str] = []
		for key, value in BYTETRACK_CONFIG.items():
			if isinstance(value, bool):
				value_str = "true" if value else "false"
			else:
				value_str = f"{value}"
			lines.append(f"{key}: {value_str}")
		content = "\n".join(lines) + "\n"
		if not BYTETRACK_CONFIG_PATH.exists() or BYTETRACK_CONFIG_PATH.read_text(encoding="utf-8") != content:
			BYTETRACK_CONFIG_PATH.write_text(content, encoding="utf-8")

	def _load_model(self) -> YOLO:
		with self._model_lock:
			if self._model is None:
				if not self._model_path.exists():
					raise FileNotFoundError(
						f"Model weights not found at {self._model_path}"  # pragma: no cover
					)
				self._model = YOLO(str(self._model_path))
		return self._model

	def reset(self) -> None:
		self._flush_behavior()
		self._track_history = defaultdict(self._new_history)
		self._heatmap = None
		model = self._load_model()
		reset_fn = getattr(model, "reset_tracker", None)
		if callable(reset_fn):
			reset_fn()
		else:
			try:
				setattr(model, "tracker", None)  # type: ignore[misc]
			except AttributeError:
				pass
		self._latest_behavior_metrics = None
		self._latest_behavior_alerts = []

	def set_trail_length(self, length: int) -> None:
		length = max(MIN_TRAIL_LENGTH, min(int(length), MAX_TRAIL_LENGTH))
		if length == self._trail_length:
			return
		self._trail_length = length
		trimmed: Dict[int, Deque[Tuple[int, int]]] = {}
		for track_id, points in self._track_history.items():
			trimmed[track_id] = deque(points, maxlen=length)
		self._track_history = defaultdict(self._new_history, trimmed)

	def set_heatmap_params(
		self,
		*,
		decay: float,
		gain: float,
		radius: int,
		blur: int,
	) -> None:
		self._heatmap_decay = max(MIN_HEATMAP_DECAY, min(decay, MAX_HEATMAP_DECAY))
		self._heatmap_gain = max(MIN_HEATMAP_GAIN, min(gain, MAX_HEATMAP_GAIN))
		radius = max(MIN_HEATMAP_RADIUS, min(int(radius), MAX_HEATMAP_RADIUS))
		self._heatmap_radius = radius
		blur = max(MIN_HEATMAP_BLUR, min(int(blur), MAX_HEATMAP_BLUR))
		if blur % 2 == 0 and blur != 0:
			blur += 1
		self._heatmap_blur = blur

	def set_behavior_context(self, context: str) -> None:
		if context:
			self._behavior_context = context

	def consume_behavior_event(self) -> Optional[Tuple[Dict[str, float], List[str]]]:
		event = self._pending_behavior_event
		self._pending_behavior_event = None
		return event

	def _flush_behavior(self) -> None:
		metrics, alerts = self._behavior.flush()
		if metrics is not None:
			self._latest_behavior_metrics = metrics
			self._latest_behavior_alerts = alerts
			self._pending_behavior_event = (metrics, alerts)

	def _draw_behavior_overlay(self, frame: np.ndarray) -> None:
		if not self._latest_behavior_metrics:
			return
		metrics = self._latest_behavior_metrics
		lines: List[str] = []
		value_map = {
			"Speed": metrics.get("mean_speed"),
			"Cluster": metrics.get("clustering"),
			"Edge": metrics.get("edge_ratio"),
			"Erratic": metrics.get("erratic_index"),
		}
		for label, value in value_map.items():
			if value is None or math.isnan(value):
				formatted = "nan"
			else:
				formatted = f"{value:.2f}"
			lines.append(f"{label}: {formatted}")
		z_map = {
			"Speed z": metrics.get("mean_speed_z"),
			"Cluster z": metrics.get("clustering_z"),
			"Edge z": metrics.get("edge_ratio_z"),
			"Erratic z": metrics.get("erratic_index_z"),
		}
		for label, value in z_map.items():
			if value is None or math.isnan(value):
				formatted = "nan"
			else:
				formatted = f"{value:.2f}"
			lines.append(f"{label}: {formatted}")
		if self._latest_behavior_alerts:
			alert_line = "Alerts: " + ", ".join(self._latest_behavior_alerts)
			lines.append(alert_line)
		y = 30
		for line in lines:
			cv2.putText(
				frame,
				line,
				(12, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(0, 0, 0),
				3,
				cv2.LINE_AA,
			)
			cv2.putText(
				frame,
				line,
				(12, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(255, 255, 255),
				1,
				cv2.LINE_AA,
			)
			y += 20

	def behavior_snapshot(self) -> Tuple[Optional[Dict[str, float]], List[str]]:
		metrics = (
			None if self._latest_behavior_metrics is None else dict(self._latest_behavior_metrics)
		)
		alerts = list(self._latest_behavior_alerts)
		return metrics, alerts

	def _update_behavior(
		self,
		frame: np.ndarray,
		boxes: Sequence[Tuple[float, float, float, float]],
		dt: float,
		timestamp: float,
		context: str,
	) -> None:
		self._behavior_context = context
		metrics, alerts = self._behavior.update(frame, boxes, dt, timestamp, context)
		if metrics is not None:
			self._latest_behavior_metrics = metrics
			self._latest_behavior_alerts = alerts
			self._pending_behavior_event = (metrics, alerts)

	def process_frame(
		self,
		frame: np.ndarray,
		conf: float,
		iou: float,
		dt: float,
		timestamp: float,
		context: Optional[str] = None,
	) -> Tuple[np.ndarray, np.ndarray]:
		self._ensure_bytetrack_config()
		model = self._load_model()
		results = model.track(
			frame,
			conf=conf,
			iou=iou,
			tracker=str(BYTETRACK_CONFIG_PATH),
			persist=True,
			verbose=False,
		)

		if not results:
			annotated_frame = frame.copy()
			self._update_behavior(
				frame,
				[],
				max(dt, 1e-6),
				timestamp,
				context or self._behavior_context,
			)
			heatmap_frame = self._update_heatmap([], annotated_frame.shape)
			return annotated_frame, heatmap_frame

		if isinstance(results, list):
			result = results[0]
		else:
			result = results
		boxes_obj = getattr(result, "boxes", None)
		if boxes_obj is not None:
			ids_tensor = getattr(boxes_obj, "id", None)
			track_ids = ids_tensor.int().tolist() if ids_tensor is not None else []
			boxes_xyxy: List[Sequence[float]] = boxes_obj.xyxy.tolist()
		else:
			track_ids = []
			boxes_xyxy = []
		behavior_boxes: List[Tuple[float, float, float, float]] = []
		for box in boxes_xyxy:
			if len(box) < 4:
				continue
			x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
			behavior_boxes.append((x1, y1, x2, y2))
		annotated_frame = frame.copy()
		centers = self._update_trails(result, track_ids, boxes_xyxy if track_ids else [])
		self._draw_detections(annotated_frame, result)
		self._draw_trails(annotated_frame)
		self._update_behavior(
			frame,
			behavior_boxes,
			max(dt, 1e-6),
			timestamp,
			context or self._behavior_context,
		)
		heatmap_frame = self._update_heatmap(centers, annotated_frame.shape)
		return annotated_frame, heatmap_frame

	def _update_trails(
		self,
		result: Any,
		track_ids: Sequence[int],
		boxes_xyxy: Sequence[Sequence[float]],
	) -> List[Tuple[int, int]]:
		if not track_ids or not boxes_xyxy:
			self._track_history.clear()
			return []

		current_ids = set()
		centers: List[Tuple[int, int]] = []
		img = getattr(result, "orig_img", None)
		shape = getattr(result, "orig_shape", None)
		height = int(img.shape[0]) if img is not None else (
			int(shape[0]) if isinstance(shape, (tuple, list)) and len(shape) >= 2 else None
		)
		width = int(img.shape[1]) if img is not None else (
			int(shape[1]) if isinstance(shape, (tuple, list)) and len(shape) >= 2 else None
		)

		for track_id, box in zip(track_ids, boxes_xyxy):
			current_ids.add(track_id)
			x1, y1, x2, y2 = box
			cx = int(round((x1 + x2) / 2))
			cy = int(round((y1 + y2) / 2))
			if width is not None:
				cx = int(np.clip(cx, 0, max(width - 1, 0)))
			if height is not None:
				cy = int(np.clip(cy, 0, max(height - 1, 0)))
			self._track_history[track_id].append((cx, cy))
			centers.append((cx, cy))

		for track_id in list(self._track_history.keys()):
			if track_id not in current_ids:
				self._track_history.pop(track_id, None)

		return centers

	def _draw_detections(self, frame: np.ndarray, result: Any) -> None:
		boxes = getattr(result, "boxes", None)
		if boxes is None or len(boxes) == 0:
			return

		xyxy = boxes.xyxy.tolist()
		ids_tensor = getattr(boxes, "id", None)
		conf_tensor = getattr(boxes, "conf", None)
		cls_tensor = getattr(boxes, "cls", None)
		names = getattr(result, "names", None)

		ids: List[int | None]
		if ids_tensor is not None:
			ids = [int(i) for i in ids_tensor.int().tolist()]
		else:
			ids = [None] * len(xyxy)

		confidences: List[float | None]
		if conf_tensor is not None:
			confidences = [float(c) for c in conf_tensor.tolist()]
		else:
			confidences = [None] * len(xyxy)

		classes: List[int | None]
		if cls_tensor is not None:
			classes = [int(c) for c in cls_tensor.int().tolist()]
		else:
			classes = [None] * len(xyxy)

		height, width = frame.shape[:2]

		for box, track_id, score, cls_id in zip(xyxy, ids, confidences, classes):
			x1, y1, x2, y2 = box
			x1 = int(np.clip(round(x1), 0, max(width - 1, 0)))
			y1 = int(np.clip(round(y1), 0, max(height - 1, 0)))
			x2 = int(np.clip(round(x2), 0, max(width - 1, 0)))
			y2 = int(np.clip(round(y2), 0, max(height - 1, 0)))
			if x2 <= x1 or y2 <= y1:
				continue
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
			parts: List[str] = []
			if track_id is not None:
				parts.append(f"ID {track_id}")
			if score is not None:
				parts.append(f"{score:.2f}")
			if names is not None and cls_id is not None:
				class_name = None
				if isinstance(names, dict):
					class_name = names.get(cls_id)
				elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
					class_name = names[cls_id]
				if class_name:
					parts.append(str(class_name))
			if parts:
				label = " ".join(parts)
				cv2.putText(
					frame,
					label,
					(x1, max(y1 - 6, 0)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(0, 0, 0),
					3,
					cv2.LINE_AA,
				)
				cv2.putText(
					frame,
					label,
					(x1, max(y1 - 6, 0)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)

	def _draw_trails(self, frame: np.ndarray) -> None:
		for points in self._track_history.values():
			if len(points) < 2:
				continue
			for start, end in zip(points, list(points)[1:]):
				cv2.line(frame, start, end, TRACK_COLOR, 2)

	def _ensure_heatmap(self, shape: Tuple[int, int, int]) -> None:
		height, width = shape[:2]
		if self._heatmap is None or self._heatmap.shape != (height, width):
			self._heatmap = np.zeros((height, width), dtype=np.float32)

	def _update_heatmap(
		self,
		centers: List[Tuple[int, int]],
		frame_shape: Tuple[int, int, int],
	) -> np.ndarray:
		self._ensure_heatmap(frame_shape)
		assert self._heatmap is not None
		self._heatmap *= self._heatmap_decay
		if centers:
			temp = np.zeros_like(self._heatmap)
			for cx, cy in centers:
				if cx < 0 or cy < 0:
					continue
				if cx >= self._heatmap.shape[1] or cy >= self._heatmap.shape[0]:
					continue
				cv2.circle(temp, (cx, cy), self._heatmap_radius, self._heatmap_gain, -1)
			self._heatmap = cv2.add(self._heatmap, temp)
		heatmap_display = self._heatmap.copy()
		if self._heatmap_blur > 0:
			ksize = self._heatmap_blur if self._heatmap_blur % 2 == 1 else self._heatmap_blur + 1
			if ksize > 1:
				heatmap_display = cv2.GaussianBlur(heatmap_display, (ksize, ksize), 0)
		min_val = float(np.min(heatmap_display))
		max_val = float(np.max(heatmap_display))
		if max_val - min_val > 1e-6:
			scaled = (heatmap_display - min_val) / (max_val - min_val)
		else:
			scaled = heatmap_display / (max_val + 1e-6)
		normalized = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
		colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
		return colored

def list_cameras(max_index: int = MAX_CAMERA_INDEX) -> List[str]:
	available = []
	for cam_index in range(max_index + 1):
		cap = cv2.VideoCapture(cam_index)
		if cap is None:
			continue
		if not cap.isOpened():
			cap.release()
			continue
		success, _ = cap.read()
		cap.release()
		if success:
			available.append(str(cam_index))

	if str(DEFAULT_CAMERA_INDEX) not in available:
		available.insert(0, str(DEFAULT_CAMERA_INDEX))

	return available


def _round_float(value: Optional[float], precision: int = 3) -> Optional[float]:
	try:
		if value is None:
			return None
		result = float(value)
		if math.isnan(result):
			return None
		return round(result, precision)
	except (TypeError, ValueError):
		return None


def _format_behavior_sections(
	metrics: Optional[Dict[str, float]],
	alerts: Sequence[str],
) -> Tuple[
	Optional[Dict[str, Any]],
	Optional[Dict[str, Any]],
	Optional[Dict[str, Any]],
	Optional[Dict[str, Any]],
]:
	if not metrics:
		return None, None, None, None

	timestamp = metrics.get("timestamp")
	context = metrics.get("context")
	alert_set = {str(alert) for alert in alerts}

	lethargy_panel: Dict[str, Any] = {
		"timestamp": timestamp,
		"context": context,
		"direct_metric": {
			"metric": "mean_speed",
			"value_mps": _round_float(metrics.get("mean_speed")),
			"note": "Lower than baseline indicates lethargy.",
		},
		"alert_driver": {
			"metric": "mean_speed_z",
			"z_score": _round_float(metrics.get("mean_speed_z")),
			"trigger": "<= -2.5 sustained",
		},
		"active_alert": "Lethargy" in alert_set,
	}

	clustering_panel: Dict[str, Any] = {
		"timestamp": timestamp,
		"context": context,
		"direct_metric": {
			"metric": "clustering",
			"value": _round_float(metrics.get("clustering")),
			"note": "Higher = more crowding.",
		},
		"support_metric": {
			"metric": "entropy",
			"value": _round_float(metrics.get("entropy")),
			"note": "Lower entropy reinforces crowding.",
		},
		"alert_drivers": {
			"clustering_z": _round_float(metrics.get("clustering_z")),
			"entropy_z": _round_float(metrics.get("entropy_z")),
			"trigger": ">= 3.0 clustering z or <= -3.0 entropy z",
		},
		"active_alert": "Clustering" in alert_set,
	}

	edge_panel: Dict[str, Any] = {
		"timestamp": timestamp,
		"context": context,
		"direct_metric": {
			"metric": "edge_ratio",
			"value": _round_float(metrics.get("edge_ratio")),
			"note": "Fraction of detections near tank walls.",
		},
		"alert_driver": {
			"metric": "edge_ratio_z",
			"z_score": _round_float(metrics.get("edge_ratio_z")),
			"trigger": ">= 2.5 sustained",
		},
		"active_alert": "EdgePacing" in alert_set,
	}

	inflow_panel: Dict[str, Any] = {
		"timestamp": timestamp,
		"context": context,
		"direct_metric": {
			"metric": "inflow_ratio",
			"value": _round_float(metrics.get("inflow_ratio")),
			"note": "Share of detections inside inflow ROI.",
		},
		"alert_driver": {
			"metric": "inflow_ratio_z",
			"z_score": _round_float(metrics.get("inflow_ratio_z")),
			"trigger": ">= 3.0 sustained",
		},
		"active_alert": "InflowMagnet" in alert_set,
	}

	return lethargy_panel, clustering_panel, edge_panel, inflow_panel


GLOBAL_TRACKER = FishTracker(MODEL_PATH)


def tracking_generator(
	camera_index: str | int,
	conf_threshold: float,
	iou_threshold: float,
	trail_length: float,
	heatmap_decay: float,
	heatmap_gain: float,
	heatmap_radius: float,
	heatmap_blur: float,
	shared_state: Dict[str, Any],
) -> Iterator[
	Tuple[
		Any,
		Any,
		Any,
		Any,
		Any,
		Any,
		Any,
		Dict[str, Any],
	]
]:
	shared_state["stop"] = False
	shared_state["behavior_metrics"] = None
	shared_state["behavior_alerts"] = []
	shared_state["behavior_sections"] = {
		"lethargy": None,
		"clustering": None,
		"edge": None,
		"inflow": None,
	}

	tracker = GLOBAL_TRACKER
	tracker.set_trail_length(int(trail_length))
	tracker.set_heatmap_params(
		decay=heatmap_decay,
		gain=heatmap_gain,
		radius=int(heatmap_radius),
		blur=int(heatmap_blur),
	)
	tracker.reset()
	raw_context = shared_state.get("context", "pre_feed")
	initial_context = raw_context if isinstance(raw_context, str) and raw_context else "pre_feed"
	tracker.set_behavior_context(initial_context)
	shared_state["context"] = initial_context

	try:
		cam_idx = int(camera_index)
	except (TypeError, ValueError):
		cam_idx = DEFAULT_CAMERA_INDEX

	cap = cv2.VideoCapture(cam_idx)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

	if not cap.isOpened():
		gr.Warning(f"Unable to open camera {camera_index}.")
		shared_state["stop"] = True
		cap.release()
		tracker.reset()
		yield (
			gr.update(value=None),
			gr.update(value=None),
			gr.update(value=None),
			gr.update(value=None),
			gr.update(value=None),
			gr.update(value=None),
			gr.update(value=None),
			shared_state,
		)
		return

	prev_time = time.time()
	smoothed_fps = 0.0
	elapsed_total = 0.0

	try:
		while not shared_state.get("stop", False):
			success, frame = cap.read()
			if not success:
				gr.Warning("Camera frame grab failed. Stopping stream.")
				break

			current_time = time.time()
			elapsed = max(current_time - prev_time, 1e-6)
			instantaneous_fps = 1.0 / elapsed
			smoothed_fps = (
				FPS_SMOOTHING * smoothed_fps
				+ (1.0 - FPS_SMOOTHING) * instantaneous_fps
			)
			prev_time = current_time
			elapsed_total += elapsed
			raw_context = shared_state.get("context", "pre_feed")
			context_value = raw_context if isinstance(raw_context, str) and raw_context else "pre_feed"
			tracker.set_behavior_context(context_value)

			annotated, heatmap = tracker.process_frame(
				frame,
				conf_threshold,
				iou_threshold,
				elapsed,
				elapsed_total,
				context_value,
			)

			behavior_event = tracker.consume_behavior_event()
			if behavior_event is not None:
				metrics_event, alerts_event = behavior_event
				shared_state["behavior_metrics"] = metrics_event
				shared_state["behavior_alerts"] = alerts_event
				if alerts_event:
					print(
						json.dumps(
							{"timestamp": metrics_event["timestamp"], "alerts": alerts_event}
						),
						flush=True,
					)

			metrics_snapshot, alerts_snapshot = tracker.behavior_snapshot()
			shared_state["behavior_metrics"] = metrics_snapshot
			shared_state["behavior_alerts"] = alerts_snapshot
			lethargy_panel, clustering_panel, edge_panel, inflow_panel = _format_behavior_sections(
				metrics_snapshot,
				alerts_snapshot,
			)
			shared_state["behavior_sections"] = {
				"lethargy": lethargy_panel,
				"clustering": clustering_panel,
				"edge": edge_panel,
				"inflow": inflow_panel,
			}
			alerts_payload = {
				"timestamp": metrics_snapshot.get("timestamp") if metrics_snapshot else None,
				"context": context_value,
				"active_alerts": alerts_snapshot,
			}

			cv2.putText(
				annotated,
				f"FPS: {smoothed_fps:4.1f}",
				(12, 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 0, 0),
				3,
				cv2.LINE_AA,
			)
			cv2.putText(
				annotated,
				f"FPS: {smoothed_fps:4.1f}",
				(12, 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)

			rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
			heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
			yield (
				rgb_frame,
				heatmap_rgb,
				lethargy_panel,
				clustering_panel,
				edge_panel,
				inflow_panel,
				alerts_payload,
				shared_state,
			)

	finally:
		shared_state["stop"] = True
		tracker.reset()
		final_event = tracker.consume_behavior_event()
		if final_event is not None:
			metrics_event, alerts_event = final_event
			shared_state["behavior_metrics"] = metrics_event
			shared_state["behavior_alerts"] = alerts_event
			if alerts_event:
				print(
					json.dumps(
						{"timestamp": metrics_event["timestamp"], "alerts": alerts_event}
					),
					flush=True,
				)
		cap.release()
	shared_state["behavior_metrics"] = None
	shared_state["behavior_alerts"] = []
	shared_state["behavior_sections"] = {
		"lethargy": None,
		"clustering": None,
		"edge": None,
		"inflow": None,
	}
	yield (
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		shared_state,
	)


def stop_tracking(shared_state: Dict[str, Any]):
	shared_state["stop"] = True
	shared_state["behavior_metrics"] = None
	shared_state["behavior_alerts"] = []
	shared_state["behavior_sections"] = {
		"lethargy": None,
		"clustering": None,
		"edge": None,
		"inflow": None,
	}
	return (
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		gr.update(value=None),
		shared_state,
	)


def refresh_camera_choices():
	choices = list_cameras()
	default_value = choices[0] if choices else str(DEFAULT_CAMERA_INDEX)
	return gr.update(choices=choices, value=default_value)


def build_interface() -> gr.Blocks:
	initial_choices = list_cameras()
	default_choice = initial_choices[0] if initial_choices else str(DEFAULT_CAMERA_INDEX)

	with gr.Blocks(title="Fish Tracker") as demo:
		shared_state = gr.State({"stop": True})

		gr.Markdown("## YOLOv11 Fish Tracker")

		with gr.Row(equal_height=True):
			output_image = gr.Image(
				label="Annotated Feed",
				streaming=True,
				interactive=False,
				scale=2,
			)
			with gr.Column(scale=1):
				with gr.Group():
					gr.Markdown("### Cameras")
					camera_dropdown = gr.Dropdown(
						label="Camera",
						choices=initial_choices,
						value=default_choice,
						interactive=True,
					)
					refresh_btn = gr.Button("Refresh Cameras")
				with gr.Group():
					gr.Markdown("### Tracking Adjustments")
					conf_slider = gr.Slider(
						minimum=MIN_CONF_THRESHOLD,
						maximum=MAX_CONF_THRESHOLD,
						value=DEFAULT_CONF_THRESHOLD,
						step=0.05,
						label="Confidence Threshold",
					)
					iou_slider = gr.Slider(
						minimum=MIN_IOU_THRESHOLD,
						maximum=MAX_IOU_THRESHOLD,
						value=DEFAULT_IOU_THRESHOLD,
						step=0.05,
						label="IoU Threshold",
					)
					trail_slider = gr.Slider(
						minimum=MIN_TRAIL_LENGTH,
						maximum=MAX_TRAIL_LENGTH,
						value=DEFAULT_TRAIL_LENGTH,
						step=1,
						label="Trail Length",
					)

		with gr.Row():
			start_btn = gr.Button("Start Tracking", variant="primary")
			stop_btn = gr.Button("Stop Tracking")

		with gr.Column():
			gr.Markdown("### Heatmap")
			heatmap_image = gr.Image(
				label="Heatmap",
				streaming=True,
				interactive=False,
			)
			with gr.Group():
				gr.Markdown("### Heatmap Adjustments")
				heatmap_decay_slider = gr.Slider(
					minimum=MIN_HEATMAP_DECAY,
					maximum=MAX_HEATMAP_DECAY,
					value=DEFAULT_HEATMAP_DECAY,
					step=0.01,
					label="Decay (0=retain less, 1=retain more)",
				)
				heatmap_gain_slider = gr.Slider(
					minimum=MIN_HEATMAP_GAIN,
					maximum=MAX_HEATMAP_GAIN,
					value=DEFAULT_HEATMAP_GAIN,
					step=0.1,
					label="Intensity Gain",
				)
				heatmap_radius_slider = gr.Slider(
					minimum=MIN_HEATMAP_RADIUS,
					maximum=MAX_HEATMAP_RADIUS,
					value=DEFAULT_HEATMAP_RADIUS,
					step=1,
					label="Point Radius",
				)
				heatmap_blur_slider = gr.Slider(
					minimum=MIN_HEATMAP_BLUR,
					maximum=MAX_HEATMAP_BLUR,
					value=DEFAULT_HEATMAP_BLUR,
					step=1,
					label="Blur Kernel Size",
				)
			gr.Markdown("### Behavior Computations")
			with gr.Group():
				behavior_lethargy_json = gr.JSON(
					label="Lethargy / Activity Drop",
					value=None,
				)
			with gr.Group():
				behavior_clustering_json = gr.JSON(
					label="Clustering / Crowding",
					value=None,
				)
			with gr.Group():
				behavior_edge_json = gr.JSON(
					label="Edge (Wall-pacing) Ratio",
					value=None,
				)
			with gr.Group():
				behavior_inflow_json = gr.JSON(
					label="Inflow Magnet",
					value=None,
				)
			with gr.Group():
				behavior_alerts_json = gr.JSON(label="Alerts Detected", value=None)

		start_btn.click(
			fn=tracking_generator,
			inputs=[
				camera_dropdown,
				conf_slider,
				iou_slider,
				trail_slider,
				heatmap_decay_slider,
				heatmap_gain_slider,
				heatmap_radius_slider,
				heatmap_blur_slider,
				shared_state,
			],
			outputs=[
				output_image,
				heatmap_image,
				behavior_lethargy_json,
				behavior_clustering_json,
				behavior_edge_json,
				behavior_inflow_json,
				behavior_alerts_json,
				shared_state,
			],
		)

		stop_btn.click(
			fn=stop_tracking,
			inputs=[shared_state],
			outputs=[
				output_image,
				heatmap_image,
				behavior_lethargy_json,
				behavior_clustering_json,
				behavior_edge_json,
				behavior_inflow_json,
				behavior_alerts_json,
				shared_state,
			],
		)

		refresh_btn.click(fn=refresh_camera_choices, outputs=camera_dropdown)

	return demo


def main() -> None:
	demo = build_interface()
	demo.queue()  # Queue avoids blocking when streaming frames.
	demo.launch()


if __name__ == "__main__":
	main()