"""Fish movement tracking application using YOLOv11 and Gradio."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Tuple

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]


MODEL_PATH = Path(__file__).with_name("best_yolo11.pt").resolve()
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


def list_cameras(max_index: int = MAX_CAMERA_INDEX) -> List[str]:
	available = []
	for cam_index in range(max_index + 1):
		cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
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
	shared_state: Dict[str, bool],
) -> Iterator[Tuple[Any, Any, Dict[str, bool]]]:
	shared_state["stop"] = False

	tracker = GLOBAL_TRACKER
	tracker.set_trail_length(int(trail_length))
	tracker.set_heatmap_params(
		decay=heatmap_decay,
		gain=heatmap_gain,
		radius=int(heatmap_radius),
		blur=int(heatmap_blur),
	)
	tracker.reset()

	try:
		cam_idx = int(camera_index)
	except (TypeError, ValueError):
		cam_idx = DEFAULT_CAMERA_INDEX

	cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

	if not cap.isOpened():
		gr.Warning(f"Unable to open camera {camera_index}.")
		shared_state["stop"] = True
		cap.release()
		tracker.reset()
		yield gr.update(value=None), gr.update(value=None), shared_state
		return

	prev_time = time.time()
	smoothed_fps = 0.0

	try:
		while not shared_state.get("stop", False):
			success, frame = cap.read()
			if not success:
				gr.Warning("Camera frame grab failed. Stopping stream.")
				break

			annotated, heatmap = tracker.process_frame(
				frame, conf_threshold, iou_threshold
			)

			current_time = time.time()
			instantaneous_fps = 1.0 / max(current_time - prev_time, 1e-6)
			smoothed_fps = (
				FPS_SMOOTHING * smoothed_fps
				+ (1.0 - FPS_SMOOTHING) * instantaneous_fps
			)
			prev_time = current_time

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
			yield rgb_frame, heatmap_rgb, shared_state

	finally:
		shared_state["stop"] = True
		tracker.reset()
		cap.release()

	yield gr.update(value=None), gr.update(value=None), shared_state


def stop_tracking(shared_state: Dict[str, bool]):
	shared_state["stop"] = True
	return (
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
			outputs=[output_image, heatmap_image, shared_state],
		)

		stop_btn.click(
			fn=stop_tracking,
			inputs=[shared_state],
			outputs=[output_image, heatmap_image, shared_state],
		)

		refresh_btn.click(fn=refresh_camera_choices, outputs=camera_dropdown)

	return demo


def main() -> None:
	demo = build_interface()
	demo.queue()  # Queue avoids blocking when streaming frames.
	demo.launch()


if __name__ == "__main__":
	main()
