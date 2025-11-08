"""Fish movement tracking application using YOLOv11 and Gradio."""

from __future__ import annotations

import os
import tempfile
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]


MODEL_PATH = Path(__file__).with_name("best_yolo11.pt").resolve()
DEFAULT_TRAIL_LENGTH = 32
MIN_TRAIL_LENGTH = 4
MAX_TRAIL_LENGTH = 128
TRACK_COLOR = (0, 255, 255)
DEFAULT_CONF_THRESHOLD = 0.5
MIN_CONF_THRESHOLD = 0.1
MAX_CONF_THRESHOLD = 1.0
DEFAULT_IOU_THRESHOLD = 0.5
MIN_IOU_THRESHOLD = 0.1
MAX_IOU_THRESHOLD = 0.9
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
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


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

	def _new_history(self) -> Deque[Tuple[int, int]]:
		return deque(maxlen=self._trail_length)

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


	def process_frame(
		self,
		frame: np.ndarray,
		conf: float,
		iou: float,
	) -> Tuple[np.ndarray, np.ndarray]:
		model = self._load_model()
		results = model.track(
			frame,
			conf=conf,
			tracker="bytetrack.yaml",
			iou=iou,
			persist=True,
			verbose=False,
		)

		if not results:
			heatmap_frame = self._update_heatmap([], frame.shape)
			return frame, heatmap_frame

		result = results[0]
		annotated_frame = result.plot()
		centers = self._update_trails(result)
		self._draw_trails(annotated_frame)
		heatmap_frame = self._update_heatmap(centers, annotated_frame.shape)
		return annotated_frame, heatmap_frame

	def _update_trails(self, result) -> List[Tuple[int, int]]:
		boxes = result.boxes
		if boxes is None or boxes.id is None:
			self._track_history.clear()
			return []

		ids = boxes.id.int().tolist()
		xyxy = boxes.xyxy.tolist()
		current_ids = set()
		centers: List[Tuple[int, int]] = []

		for track_id, box in zip(ids, xyxy):
			current_ids.add(track_id)
			x1, y1, x2, y2 = box
			cx = int((x1 + x2) / 2)
			cy = int((y1 + y2) / 2)
			self._track_history[track_id].append((cx, cy))
			centers.append((cx, cy))

		# Drop stale trajectories so the overlay stays tidy.
		for track_id in list(self._track_history.keys()):
			if track_id not in current_ids:
				self._track_history.pop(track_id, None)

		return centers

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

GLOBAL_TRACKER = FishTracker(MODEL_PATH)


def _coerce_path(media_input: Any) -> Path:
	if media_input is None:
		raise gr.Error("Please upload a video file before processing.")
	if isinstance(media_input, Path):
		path = media_input
	elif isinstance(media_input, str):
		path = Path(media_input)
	elif isinstance(media_input, dict):
		candidate = media_input.get("name") or media_input.get("path")
		if not candidate:
			raise gr.Error("Unable to determine the uploaded file path.")
		path = Path(candidate)
	else:
		candidate = getattr(media_input, "name", None)
		if not candidate:
			raise gr.Error("Unable to determine the uploaded file path.")
		path = Path(candidate)
	if not path.exists():
		raise gr.Error("Uploaded video file could not be found on disk.")
	return path


def process_uploaded_media(
	media_input: Any,
	conf_threshold: float,
	iou_threshold: float,
	trail_length: float,
	heatmap_decay: float,
	heatmap_gain: float,
	heatmap_radius: float,
	heatmap_blur: float,
) -> Tuple[str, np.ndarray]:
	tracker = GLOBAL_TRACKER
	tracker.set_trail_length(int(trail_length))
	tracker.set_heatmap_params(
		decay=heatmap_decay,
		gain=heatmap_gain,
		radius=int(heatmap_radius),
		blur=int(heatmap_blur),
	)
	tracker.reset()

	media_path = _coerce_path(media_input)
	suffix = media_path.suffix.lower()
	if suffix not in VIDEO_EXTENSIONS:
		supported = ", ".join(sorted(VIDEO_EXTENSIONS))
		raise gr.Error(
			f"Unsupported video format '{suffix}'. Please upload one of: {supported}."
		)

	cap: cv2.VideoCapture | None = None
	writer: cv2.VideoWriter | None = None
	output_path = ""
	last_heatmap: np.ndarray | None = None

	try:
		cap = cv2.VideoCapture(str(media_path))
		if not cap.isOpened():
			raise gr.Error("Unable to open the uploaded video.")

		fps = cap.get(cv2.CAP_PROP_FPS)
		if not np.isfinite(fps) or fps <= 1e-3:
			fps = 30.0

		success, frame = cap.read()
		if not success or frame is None:
			raise gr.Error("Uploaded video contains no readable frames.")

		annotated_frame, last_heatmap = tracker.process_frame(
			frame, conf_threshold, iou_threshold
		)
		height, width = annotated_frame.shape[:2]
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		fd, temp_path = tempfile.mkstemp(suffix=".mp4")
		os.close(fd)
		output_path = temp_path
		writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
		if not writer.isOpened():
			raise gr.Error("Unable to create the annotated video on disk.")

		writer.write(annotated_frame)

		while True:
			success, frame = cap.read()
			if not success or frame is None:
				break
			annotated_frame, last_heatmap = tracker.process_frame(
				frame, conf_threshold, iou_threshold
			)
			writer.write(annotated_frame)

	finally:
		if cap is not None:
			cap.release()
		if writer is not None:
			writer.release()
		tracker.reset()

	if not output_path:
		raise gr.Error("Failed to produce the annotated video output.")
	if last_heatmap is None:
		raise gr.Error("Unable to compute a heatmap for the uploaded video.")

	heatmap_rgb = cv2.cvtColor(last_heatmap, cv2.COLOR_BGR2RGB)
	return output_path, heatmap_rgb


def build_interface() -> gr.Blocks:
	with gr.Blocks(title="Fish Tracker") as demo:
		gr.Markdown("## YOLOv11 Fish Tracker")
		gr.Markdown("Upload a video to generate annotated detections and a cumulative heatmap.")

		with gr.Row(equal_height=True):
			video_input = gr.Video(
				label="Uploaded Video",
				sources=["upload"],
				type="filepath",
			)
			with gr.Column(scale=1):
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
			with gr.Column(scale=1):
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

		run_btn = gr.Button("Process Video", variant="primary")

		with gr.Row(equal_height=True):
			annotated_video = gr.Video(
				label="Annotated Video",
				autoplay=False,
				interactive=False,
			)
			heatmap_image = gr.Image(
				label="Final Heatmap",
				interactive=False,
			)

		run_btn.click(
			fn=process_uploaded_media,
			inputs=[
				video_input,
				conf_slider,
				iou_slider,
				trail_slider,
				heatmap_decay_slider,
				heatmap_gain_slider,
				heatmap_radius_slider,
				heatmap_blur_slider,
			],
			outputs=[annotated_video, heatmap_image],
		)

	return demo


def main() -> None:
	demo = build_interface()
	demo.queue()  # Queue avoids blocking when streaming frames.
	demo.launch()


if __name__ == "__main__":
	main()
