"""Heatmap utilities for sturgeon monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

from .tracking import Track

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position


_DEFAULT_COLORMAP = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET


@dataclass
class LiveHeatmapConfig:
    """Configuration for live heatmap rendering."""

    decay: float = 0.95
    blur_kernel_size: int = 25
    overlay_alpha: float = 0.5
    colormap: int = _DEFAULT_COLORMAP


class LiveHeatmap:
    """Maintains a rolling heatmap of track centroids for live display."""

    def __init__(self, width: int, height: int, config: LiveHeatmapConfig | None = None):
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Heatmap width and height must be positive")
        self.config = config or LiveHeatmapConfig()
        self._accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def reset(self) -> None:
        self._accumulator.fill(0.0)

    def update(self, tracks: Iterable[Track]) -> None:
        for track in tracks:
            x, y = track.centroid
            xi = max(0, min(self.width - 1, int(round(x))))
            yi = max(0, min(self.height - 1, int(round(y))))
            self._accumulator[yi, xi] += 1.0
        if 0.0 < self.config.decay < 1.0:
            self._accumulator *= self.config.decay

    def render(self) -> np.ndarray:
        if not np.any(self._accumulator):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        heatmap = self._accumulator.copy()
        kernel = self._validated_kernel()
        if kernel is not None:
            heatmap = cv2.GaussianBlur(heatmap, (kernel, kernel), 0)
        min_val = float(heatmap.min())
        max_val = float(heatmap.max())
        if max_val - min_val > 1e-6:
            scale = 255.0 / (max_val - min_val)
            scaled = (heatmap - min_val) * scale
        else:
            scaled = np.zeros_like(heatmap)
        normalized = scaled.astype(np.uint8)
        colored = cv2.applyColorMap(normalized, self.config.colormap)
        return colored

    def overlay(self, frame: np.ndarray, alpha: float | None = None) -> np.ndarray:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame dimensions must match heatmap dimensions for overlay")
        overlay_alpha = self.config.overlay_alpha if alpha is None else float(alpha)
        overlay_alpha = max(0.0, min(1.0, overlay_alpha))
        heatmap = self.render()
        blended = cv2.addWeighted(heatmap, overlay_alpha, frame, 1.0 - overlay_alpha, 0.0)
        return blended

    def _validated_kernel(self) -> int | None:
        kernel = int(self.config.blur_kernel_size)
        if kernel <= 1:
            return None
        if kernel % 2 == 0:
            kernel += 1
        return kernel


@dataclass
class IntervalHeatmapConfig:
    """Configuration for saving time-windowed analytics plots."""

    interval_seconds: float = 900.0
    heatmap_bins: Tuple[int, int] | None = None
    cmap: str = "inferno"
    scatter_size: int = 20
    scatter_alpha: float = 0.6


class HeatmapIntervalSaver:
    """Collects centroid positions and writes periodic heatmap and scatter plots."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        video_stem: str,
        output_dir: Path | str,
        config: IntervalHeatmapConfig | None = None,
    ):
        if fps <= 0:
            raise ValueError("FPS must be positive")
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.video_stem = video_stem
        self.output_dir = Path(output_dir)
        self.config = config or IntervalHeatmapConfig()
        self._heatmap_dir = self.output_dir / "heatmaps"
        self._scatter_dir = self.output_dir / "scatter_plots"
        self._heatmap_dir.mkdir(parents=True, exist_ok=True)
        self._scatter_dir.mkdir(parents=True, exist_ok=True)
        self._window_index = 1
        self._window_start_frame: int | None = None
        self._window_start_time: float | None = None
        self._last_frame_index: int | None = None
        self._positions: List[Tuple[float, float]] = []
        self._saved_windows: List[Tuple[Path, Path]] = []

    @property
    def saved_windows(self) -> Sequence[Tuple[Path, Path]]:
        return tuple(self._saved_windows)

    def observe(self, tracks: Iterable[Track], frame_index: int) -> None:
        if self._window_start_frame is None:
            self._window_start_frame = frame_index
            self._window_start_time = frame_index / self.fps
        self._last_frame_index = frame_index
        for track in tracks:
            x, y = track.centroid
            if 0.0 <= x < self.width and 0.0 <= y < self.height:
                self._positions.append((x, y))
        if self._window_elapsed_seconds(frame_index) >= self.config.interval_seconds:
            self._finalize_window(partial=False)

    def finalize(self) -> None:
        if self._positions:
            self._finalize_window(partial=True)

    # Internal helpers -------------------------------------------------

    def _window_elapsed_seconds(self, frame_index: int) -> float:
        if self._window_start_time is None:
            return 0.0
        current_time = frame_index / self.fps
        return max(0.0, current_time - self._window_start_time)

    def _finalize_window(self, partial: bool) -> None:
        if not self._positions:
            self._advance_window()
            return
        heatmap_path = self._heatmap_dir / self._make_filename(partial, suffix="heatmap")
        scatter_path = self._scatter_dir / self._make_filename(partial, suffix="scatter")
        self._write_heatmap(heatmap_path)
        self._write_scatter(scatter_path)
        self._saved_windows.append((heatmap_path, scatter_path))
        self._advance_window()

    def _make_filename(self, partial: bool, suffix: str) -> str:
        start_label = self._format_time(self._window_start_time or 0.0)
        end_seconds = (self._last_frame_index or 0) / self.fps
        end_label = self._format_time(end_seconds)
        partial_tag = "_partial" if partial else ""
        return f"{self.video_stem}_window{self._window_index:03d}{partial_tag}_{start_label}_to_{end_label}_{suffix}.png"

    def _format_time(self, seconds: float) -> str:
        total_seconds = int(seconds)
        return str(timedelta(seconds=total_seconds)).replace(":", "-")

    def _advance_window(self) -> None:
        self._window_index += 1
        self._positions.clear()
        if self._last_frame_index is None:
            self._window_start_frame = None
            self._window_start_time = None
            return
        next_start_frame = self._last_frame_index + 1
        self._window_start_frame = next_start_frame
        self._window_start_time = next_start_frame / self.fps

    def _resolve_bins(self) -> Tuple[int, int]:
        if self.config.heatmap_bins:
            bins_x, bins_y = self.config.heatmap_bins
        else:
            bins_x = max(10, self.width // 20)
            bins_y = max(10, self.height // 20)
        return max(1, bins_x), max(1, bins_y)

    def _write_heatmap(self, path: Path) -> None:
        xs, ys = self._positions_array()
        bins_x, bins_y = self._resolve_bins()
        heat, _, _ = np.histogram2d(ys, xs, bins=[bins_y, bins_x], range=[[0, self.height], [0, self.width]])
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        extent = (0, self.width, 0, self.height)
        img = ax.imshow(heat, origin="lower", cmap=self.config.cmap, extent=extent, aspect="auto")
        ax.set_title("Sturgeon density heatmap")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        fig.colorbar(img, ax=ax, label="Detections")
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _write_scatter(self, path: Path) -> None:
        xs, ys = self._positions_array()
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.scatter(xs, ys, s=self.config.scatter_size, alpha=self.config.scatter_alpha, c="cyan", edgecolors="none")
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title("Sturgeon positions")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        ax.invert_yaxis()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _positions_array(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._positions:
            raise ValueError("No positions recorded for the current window")
        data = np.array(self._positions, dtype=np.float32)
        xs = data[:, 0]
        ys = data[:, 1]
        return xs, ys
