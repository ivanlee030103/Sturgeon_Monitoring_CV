"""Behavior analytics without ID tracking using OpenCV optical flow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def preprocess(frame_bgr: np.ndarray, tank_mask: Optional[np.ndarray] = None) -> np.ndarray:
	"""Convert frame to masked, blurred grayscale for optical flow."""
	gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
	if tank_mask is not None:
		if tank_mask.shape != gray.shape:
			mask_resized = cv2.resize(tank_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
		else:
			mask_resized = tank_mask
		gray = cv2.bitwise_and(gray, gray, mask=mask_resized)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	return gray


def compute_flow(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> np.ndarray:
	"""Compute dense Farnebäck optical flow between frames."""
	if prev_gray is None or prev_gray.shape != gray.shape:
		return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
	return cv2.calcOpticalFlowFarneback(
		prev_gray,
		gray,
		None,  # type: ignore[arg-type]
		pyr_scale=0.5,
		levels=4,
		winsize=27,
		iterations=4,
		poly_n=7,
		poly_sigma=1.2,
		flags=0,
	)


def speeds_from_boxes(
	flow_mag: np.ndarray,
	boxes: Sequence[Tuple[float, float, float, float]],
	meters_per_pixel: float,
	dt: float,
) -> np.ndarray:
	"""Median flow magnitude per box converted to m/s."""
	if dt <= 0.0 or not boxes:
		return np.zeros(len(boxes), dtype=np.float32)

	h, w = flow_mag.shape
	speeds = np.zeros(len(boxes), dtype=np.float32)
	scale = meters_per_pixel / max(dt, 1e-6)
	for idx, (x1, y1, x2, y2) in enumerate(boxes):
		x1i = int(max(0, min(w - 1, math.floor(x1))))
		y1i = int(max(0, min(h - 1, math.floor(y1))))
		x2i = int(max(0, min(w, math.ceil(x2))))
		y2i = int(max(0, min(h, math.ceil(y2))))
		if x2i <= x1i or y2i <= y1i:
			continue
		patch = flow_mag[y1i:y2i, x1i:x2i]
		if patch.size == 0:
			continue
		speeds[idx] = float(np.median(patch)) * scale
	return speeds


def occupancy_update(
	grid: np.ndarray,
	centroids: np.ndarray,
	frame_shape: Tuple[int, int, int],
) -> np.ndarray:
	"""Accumulate centroid counts into a 2D occupancy grid."""
	if centroids.size == 0:
		return grid
	rows, cols = grid.shape
	h, w = frame_shape[:2]
	x_norm = np.clip((centroids[:, 0] / max(w, 1e-6)) * cols, 0, cols - 1)
	y_norm = np.clip((centroids[:, 1] / max(h, 1e-6)) * rows, 0, rows - 1)
	indices = np.stack((y_norm.astype(int), x_norm.astype(int)))
	np.add.at(grid, (indices[0], indices[1]), 1)
	return grid


def clustering_entropy(grid: np.ndarray) -> Tuple[float, float]:
	"""Return clustering mass ratio and normalized entropy for occupancy grid."""
	flat = grid.reshape(-1)
	total = float(flat.sum())
	if total <= 1e-6:
		return 0.0, 0.0
	cells = flat.size
	top_k = max(1, int(math.ceil(cells * 0.1)))
	top_mass = np.partition(flat, cells - top_k)[-top_k:].sum()
	clustering_ratio = float(top_mass / total)
	prob = flat / total
	nz = prob > 0
	entropy = float(-(prob[nz] * np.log(prob[nz])).sum())
	norm_entropy = entropy / math.log(cells)
	return clustering_ratio, float(np.clip(norm_entropy, 0.0, 1.0))


def edge_ratio(
	centroids: np.ndarray,
	frame_shape: Tuple[int, int, int],
	band_fraction: float = 0.12,
) -> float:
	"""Compute fraction of centroids within tank perimeter band."""
	if centroids.size == 0:
		return 0.0
	h, w = frame_shape[:2]
	band = band_fraction * min(h, w)
	distances = np.stack(
		[
			centroids[:, 0],
			w - centroids[:, 0],
			centroids[:, 1],
			h - centroids[:, 1],
		],
		axis=1,
	)
	min_dist = distances.min(axis=1)
	return float((min_dist <= band).sum()) / centroids.shape[0]


def roi_ratio(centroids: np.ndarray, roi: Optional[Sequence[Tuple[float, float]]]) -> float:
	"""Compute fraction of centroids inside a supplied polygon ROI."""
	if centroids.size == 0 or roi is None or len(roi) < 3:
		return 0.0
	contour = np.asarray(roi, dtype=np.float32)
	inside = 0
	for cx, cy in centroids:
		if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0:
			inside += 1
	return float(inside) / centroids.shape[0]


def zscores(
	values: Dict[str, float],
	stats: "RollingStats",
	warmup_seconds: int,
) -> Dict[str, float]:
	"""Compute z-scores for metrics using context-specific rolling stats."""
	ready = stats.total_count >= warmup_seconds
	result: Dict[str, float] = {}
	for key, value in values.items():
		if not ready:
			result[key] = float("nan")
			continue
		std = stats.std(key)
		if math.isnan(std) or std < 1e-6:
			result[key] = 0.0
			continue
		result[key] = (value - stats.mean(key)) / std
	return result


class DebouncePersist:
	"""Accumulate time while a condition stays high and emit when it persists."""

	def __init__(self, threshold_seconds: float) -> None:
		self.threshold = float(threshold_seconds)
		self._timer = 0.0
		self._active = False

	def update(self, condition: bool, dt: float) -> bool:
		if condition:
			self._timer += dt
			if self._timer >= self.threshold:
				self._active = True
		else:
			self.reset()
		return self._active

	def reset(self) -> None:
		self._timer = 0.0
		self._active = False


class RollingStats:
	"""Per-metric Welford accumulators for a single behavioral context."""

	def __init__(self) -> None:
		self._stats: Dict[str, Tuple[int, float, float]] = {}
		self.total_count = 0

	def update(self, values: Dict[str, float]) -> None:
		self.total_count += 1
		for key, value in values.items():
			if math.isnan(value):
				continue
			n, mean, m2 = self._stats.get(key, (0, 0.0, 0.0))
			n += 1
			delta = value - mean
			mean += delta / n
			delta2 = value - mean
			m2 += delta * delta2
			self._stats[key] = (n, mean, m2)

	def mean(self, key: str) -> float:
		return self._stats.get(key, (0, 0.0, 0.0))[1]

	def std(self, key: str) -> float:
		n, _, m2 = self._stats.get(key, (0, 0.0, 0.0))
		if n < 2:
			return float("nan")
		return math.sqrt(m2 / (n - 1))


@dataclass
class _SecondAccumulator:
	"""Per-second buffering for aggregated metrics."""

	timestamp: int
	context: str
	speeds: List[float]
	centroids: List[np.ndarray]
	occupancy: np.ndarray
	frame_count: int
	dt_total: float


def _empty_accumulator(timestamp: int, context: str, grid_shape: Tuple[int, int]) -> _SecondAccumulator:
	return _SecondAccumulator(
		timestamp=timestamp,
		context=context,
		speeds=[],
		centroids=[],
		occupancy=np.zeros(grid_shape, dtype=np.float32),
		frame_count=0,
		dt_total=0.0,
	)


@dataclass
class DetectionRecord:
	boxes: List[Tuple[float, float, float, float]]
	timestamp: float
	context: str


class BehaviorLite:
	"""Compute ID-free behavioral metrics and persistent alerts from video frames."""

	GRID_SHAPE = (20, 20)

	def __init__(
		self,
		meters_per_pixel: float,
		tank_mask: Optional[np.ndarray] = None,
		inflow_roi: Optional[Sequence[Tuple[float, float]]] = None,
		warmup_seconds: int = 600,
	) -> None:
		self._meters_per_pixel = float(meters_per_pixel)
		self._tank_mask = None if tank_mask is None else tank_mask.astype(np.uint8)
		self._inflow_roi: Optional[List[Tuple[float, float]]] = (
			None if inflow_roi is None else [(float(x), float(y)) for x, y in inflow_roi]
		)
		self._warmup = warmup_seconds
		self._prev_gray: Optional[np.ndarray] = None
		self._current_second: Optional[int] = None
		self._accumulator: Optional[_SecondAccumulator] = None
		self._frame_shape: Optional[Tuple[int, int, int]] = None
		self._context_stats: Dict[str, RollingStats] = {
			"pre_feed": RollingStats(),
			"feeding": RollingStats(),
			"post_feed": RollingStats(),
			"night": RollingStats(),
		}
		self._debouncers: Dict[str, DebouncePersist] = {
			"Lethargy": DebouncePersist(150.0),
			"Clustering": DebouncePersist(90.0),
			"EdgePacing": DebouncePersist(60.0),
			"InflowMagnet": DebouncePersist(60.0),
			"Erratic": DebouncePersist(45.0),
			"NoFeedResponse": DebouncePersist(60.0),
		}

	def update(
		self,
		frame_bgr: np.ndarray,
		yolo_boxes: Sequence[Tuple[float, float, float, float]],
		dt: float,
		timestamp: float,
		context: str,
		meters_per_pixel: Optional[float] = None,
	) -> Tuple[Optional[Dict[str, float]], List[str]]:
		"""Process a frame, returning per-second metrics when a second completes."""
		if meters_per_pixel is not None:
			self._meters_per_pixel = float(meters_per_pixel)

		self._frame_shape = frame_bgr.shape
		context = context if context in self._context_stats else "pre_feed"

		gray = preprocess(frame_bgr, self._tank_mask)
		flow = compute_flow(self._prev_gray, gray)
		flow_mag = cv2.magnitude(flow[..., 0], flow[..., 1])

		filtered_boxes, centroids = self._filter_boxes(yolo_boxes)
		speeds = speeds_from_boxes(flow_mag, filtered_boxes, self._meters_per_pixel, dt)

		metrics, alerts = None, []
		second = int(timestamp)
		if self._current_second is None:
			self._current_second = second
			self._accumulator = _empty_accumulator(second, context, self.GRID_SHAPE)
		elif second != self._current_second:
			if self._accumulator is not None and self._accumulator.frame_count > 0:
				metrics, alerts = self._finalize_second()
			self._current_second = second
			self._accumulator = _empty_accumulator(second, context, self.GRID_SHAPE)

		self._accumulate_frame(context, dt, speeds, centroids)
		self._prev_gray = gray
		return metrics, alerts

	def flush(self) -> Tuple[Optional[Dict[str, float]], List[str]]:
		"""Emit metrics for the final partial second, if present."""
		if self._accumulator is None or self._accumulator.frame_count == 0:
			return None, []
		metrics, alerts = self._finalize_second()
		self._accumulator = None
		self._current_second = None
		return metrics, alerts

	def _filter_boxes(
		self,
		boxes: Sequence[Tuple[float, float, float, float]],
	) -> Tuple[List[Tuple[float, float, float, float]], np.ndarray]:
		if not boxes:
			return [], np.empty((0, 2), dtype=np.float32)
		filtered: List[Tuple[float, float, float, float]] = []
		centroid_list: List[Tuple[float, float]] = []
		mask = self._tank_mask
		for x1, y1, x2, y2 in boxes:
			cx = 0.5 * (x1 + x2)
			cy = 0.5 * (y1 + y2)
			if mask is not None:
				xi = int(np.clip(round(cx), 0, mask.shape[1] - 1))
				yi = int(np.clip(round(cy), 0, mask.shape[0] - 1))
				if mask[yi, xi] == 0:
					continue
			filtered.append((x1, y1, x2, y2))
			centroid_list.append((cx, cy))
		if not filtered:
			return [], np.empty((0, 2), dtype=np.float32)
		return filtered, np.asarray(centroid_list, dtype=np.float32)

	def _accumulate_frame(
		self,
		context: str,
		dt: float,
		speeds: np.ndarray,
		centroids: np.ndarray,
	) -> None:
		if self._accumulator is None:
			return
		self._accumulator.context = context
		self._accumulator.frame_count += 1
		self._accumulator.dt_total += max(dt, 0.0)
		if speeds.size:
			self._accumulator.speeds.extend(speeds.tolist())
		if centroids.size:
			self._accumulator.centroids.append(centroids)
			self._accumulator.occupancy = occupancy_update(
				self._accumulator.occupancy,
				centroids,
				self._frame_shape if self._frame_shape is not None else (1, 1, 1),
			)

	def _finalize_second(self) -> Tuple[Dict[str, float], List[str]]:
		acc = self._accumulator
		assert acc is not None
		frame_shape = self._frame_shape if self._frame_shape is not None else (1, 1, 1)
		speeds = np.asarray(acc.speeds, dtype=np.float32)
		centroids = (
			np.vstack(acc.centroids) if acc.centroids else np.empty((0, 2), dtype=np.float32)
		)
		mean_speed = float(np.mean(speeds)) if speeds.size else 0.0
		std_speed = float(np.std(speeds)) if speeds.size else 0.0
		erratic = std_speed / (mean_speed + 1e-6)
		clustering, entropy_norm = clustering_entropy(acc.occupancy)
		edge = edge_ratio(centroids, frame_shape)
		inflow = roi_ratio(centroids, self._inflow_roi)

		values = {
			"mean_speed": mean_speed,
			"clustering": clustering,
			"entropy": entropy_norm,
			"edge_ratio": edge,
			"inflow_ratio": inflow,
			"erratic_index": erratic,
		}

		stats = self._context_stats[acc.context]
		z = zscores(values, stats, self._warmup)
		stats.update(values)
		alerts = self._evaluate_alerts(acc.context, z)

		metrics = {
			"timestamp": acc.timestamp,
			"context": acc.context,
			"mean_speed": mean_speed,
			"clustering": clustering,
			"entropy": entropy_norm,
			"edge_ratio": edge,
			"inflow_ratio": inflow,
			"erratic_index": erratic,
			"mean_speed_z": z.get("mean_speed", float("nan")),
			"clustering_z": z.get("clustering", float("nan")),
			"entropy_z": z.get("entropy", float("nan")),
			"edge_ratio_z": z.get("edge_ratio", float("nan")),
			"inflow_ratio_z": z.get("inflow_ratio", float("nan")),
			"erratic_index_z": z.get("erratic_index", float("nan")),
		}
		return metrics, alerts

	def _evaluate_alerts(self, context: str, z: Dict[str, float]) -> List[str]:
		alerts: List[str] = []
		z_mean = z.get("mean_speed", float("nan"))
		z_cluster = z.get("clustering", float("nan"))
		z_edge = z.get("edge_ratio", float("nan"))
		z_inflow = z.get("inflow_ratio", float("nan"))
		z_erratic = z.get("erratic_index", float("nan"))

		if context != "night" and not math.isnan(z_mean):
			if self._debouncers["Lethargy"].update(z_mean <= -2.5, 1.0):
				alerts.append("Lethargy")
		else:
			self._debouncers["Lethargy"].reset()

		if not math.isnan(z_cluster) and self._debouncers["Clustering"].update(z_cluster >= 3.0, 1.0):
			alerts.append("Clustering")
		if not math.isnan(z_edge) and self._debouncers["EdgePacing"].update(z_edge >= 2.5, 1.0):
			alerts.append("EdgePacing")
		if not math.isnan(z_inflow) and self._debouncers["InflowMagnet"].update(z_inflow >= 3.0, 1.0):
			alerts.append("InflowMagnet")
		if not math.isnan(z_erratic) and self._debouncers["Erratic"].update(z_erratic >= 3.0, 1.0):
			alerts.append("Erratic")

		if context == "feeding" and not math.isnan(z_mean):
			if self._debouncers["NoFeedResponse"].update(z_mean < 1.5, 1.0):
				alerts.append("NoFeedResponse")
		else:
			self._debouncers["NoFeedResponse"].reset()

		return alerts


def _load_mask(path: Optional[Path], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
	if path is None:
		return None
	mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise FileNotFoundError(f"Tank mask not found: {path}")
	if mask.shape != target_shape:
		mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
	return (mask > 0).astype(np.uint8)


def _load_roi(path: Optional[Path]) -> Optional[List[Tuple[float, float]]]:
	if path is None:
		return None
	with path.open("r", encoding="utf-8") as f:
		points = json.load(f)
	cleaned: List[Tuple[float, float]] = []
	for pt in points:
		x, y = (float(pt[0]), float(pt[1]))
		cleaned.append((x, y))
	return cleaned


def _load_detections(path: Path) -> Dict[int, DetectionRecord]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	frames: Dict[int, DetectionRecord] = {}
	for entry in data:
		frame_idx = int(entry.get("frame", len(frames)))
		boxes_raw = entry.get("boxes", [])
		boxes: List[Tuple[float, float, float, float]] = []
		for box in boxes_raw:
			if len(box) < 4:
				continue
			x1, y1, x2, y2 = map(float, box[:4])
			boxes.append((x1, y1, x2, y2))
		timestamp = float(entry.get("timestamp", frame_idx))
		context = str(entry.get("context", "pre_feed"))
		frames[frame_idx] = DetectionRecord(boxes=boxes, timestamp=timestamp, context=context)
	return frames


def run_cli(args: argparse.Namespace) -> None:
	video_path = Path(args.video)
	detect_path = Path(args.detections)
	detections = _load_detections(detect_path)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Unable to open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
	dt = 1.0 / fps if fps > 0 else 0.1

	ret, frame = cap.read()
	if not ret:
		raise RuntimeError("Video is empty")

	tank_mask = _load_mask(Path(args.mask) if args.mask else None, (frame.shape[0], frame.shape[1]))
	inflow_roi = _load_roi(Path(args.roi) if args.roi else None)

	behavior = BehaviorLite(
		meters_per_pixel=args.meters_per_pixel,
		tank_mask=tank_mask,
		inflow_roi=inflow_roi,
	)

	metrics_path = Path(args.metrics)
	with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
		fieldnames = [
			"timestamp",
			"context",
			"mean_speed",
			"clustering",
			"entropy",
			"edge_ratio",
			"inflow_ratio",
			"erratic_index",
			"mean_speed_z",
			"clustering_z",
			"entropy_z",
			"edge_ratio_z",
			"inflow_ratio_z",
			"erratic_index_z",
		]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()

		frame_idx = 0
		while True:
			if frame_idx > 0:
				ret, frame = cap.read()
				if not ret:
					break

			record = detections.get(frame_idx)
			if record is not None:
				boxes = record.boxes
				timestamp = record.timestamp
				context = record.context
			else:
				boxes = []
				timestamp = frame_idx * dt
				context = "pre_feed"

			metrics, alerts = behavior.update(frame, boxes, dt, timestamp, context)
			if metrics is not None:
				writer.writerow(metrics)
				if alerts:
					print(json.dumps({"timestamp": metrics["timestamp"], "alerts": alerts}), file=sys.stdout, flush=True)

			frame_idx += 1

		# flush remaining data
		metrics, alerts = behavior.flush()
		if metrics is not None:
			writer.writerow(metrics)
			if alerts:
				print(json.dumps({"timestamp": metrics["timestamp"], "alerts": alerts}), file=sys.stdout, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="Behavior metric extractor")
	parser.add_argument("video", help="Path to input video file")
	parser.add_argument("detections", help="Path to JSON detection file")
	parser.add_argument("metrics", help="Output CSV path for per-second metrics")
	parser.add_argument("--mask", help="Optional tank mask image (grayscale)")
	parser.add_argument("--roi", help="Optional JSON polygon for inflow ROI")
	parser.add_argument(
		"--meters-per-pixel",
		type=float,
		default=0.01,
		help="Scene scale in meters per pixel",
	)
	args = parser.parse_args(argv)
	run_cli(args)


if __name__ == "__main__":
	main()
