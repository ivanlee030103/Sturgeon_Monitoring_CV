"""Fish movement tracking application using YOLOv11 and Gradio."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]


MODEL_PATH = Path(__file__).with_name("best_yolo11.pt").resolve()
DEFAULT_CAMERA_INDEX = 0
MAX_CAMERA_INDEX = 8
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_SMOOTHING = 0.9
DEFAULT_CONF_THRESHOLD = 0.5
MIN_CONF_THRESHOLD = 0.1
MAX_CONF_THRESHOLD = 1.0
DEFAULT_IOU_THRESHOLD = 0.5
MIN_IOU_THRESHOLD = 0.1
MAX_IOU_THRESHOLD = 0.9



class FishTracker:
	"""Wrapper around a YOLOv11 model with simple trajectory drawing."""

	def __init__(self, model_path: Path) -> None:
		self._model_path = model_path
		self._model: YOLO | None = None
		self._model_lock = threading.Lock()

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
		model = self._load_model()
		reset_fn = getattr(model, "reset_tracker", None)
		if callable(reset_fn):
			reset_fn()
		else:
			try:
				setattr(model, "tracker", None)  # type: ignore[misc]
			except AttributeError:
				pass

	def process_frame(
		self,
		frame: np.ndarray,
		conf: float,
		iou: float,
	) -> np.ndarray:
		model = self._load_model()
		results = model.predict(
			frame,
			conf=conf,
			iou=iou,
			verbose=False,
		)

		if not results:
			return frame

		result = results[0]
		annotated_frame = result.plot()
		return annotated_frame


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
	shared_state: Dict[str, bool],
) -> Iterator[Tuple[Any, Dict[str, bool]]]:
	shared_state["stop"] = False

	tracker = GLOBAL_TRACKER
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
		yield gr.update(value=None), shared_state
		return

	prev_time = time.time()
	smoothed_fps = 0.0

	try:
		while not shared_state.get("stop", False):
			success, frame = cap.read()
			if not success:
				gr.Warning("Camera frame grab failed. Stopping stream.")
				break

			annotated = tracker.process_frame(
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
			yield rgb_frame, shared_state

	finally:
		shared_state["stop"] = True
		tracker.reset()
		cap.release()

	yield gr.update(value=None), shared_state


def stop_tracking(shared_state: Dict[str, bool]):
	shared_state["stop"] = True
	return (
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

		gr.Markdown("## YOLOv11 Fish Detector")

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
					gr.Markdown("### Detection Adjustments")
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

		with gr.Row():
			start_btn = gr.Button("Start Detection", variant="primary")
			stop_btn = gr.Button("Stop Detection")

		start_btn.click(
			fn=tracking_generator,
			inputs=[
				camera_dropdown,
				conf_slider,
				iou_slider,
				shared_state,
			],
			outputs=[output_image, shared_state],
		)

		stop_btn.click(
			fn=stop_tracking,
			inputs=[shared_state],
			outputs=[output_image, shared_state],
		)

		refresh_btn.click(fn=refresh_camera_choices, outputs=camera_dropdown)

	return demo


def main() -> None:
	demo = build_interface()
	demo.queue()  # Queue avoids blocking when streaming frames.
	demo.launch()


if __name__ == "__main__":
	main()
