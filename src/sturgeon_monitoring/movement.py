"""Movement analysis utilities for tracked sturgeon."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - handled in to_dataframe
    pd = None

from .config import MovementConfig
from .tracking import Track


@dataclass
class TrackSummary:
    """Summary statistics for a single track."""

    track_id: int
    path_length_m: float
    avg_speed_m_per_s: float
    total_frames: int

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "track_id": self.track_id,
            "path_length_m": self.path_length_m,
            "avg_speed_m_per_s": self.avg_speed_m_per_s,
            "total_frames": self.total_frames,
        }


class MovementAnalyzer:
    """Accumulates tracks to compute movement statistics."""

    def __init__(self, config: MovementConfig | None = None, fps: float = 30.0):
        self.config = config or MovementConfig()
        self.fps = fps
        self._tracks: Dict[int, Dict[int, Tuple[float, float]]] = {}

    def update(self, tracks: Iterable[Track]) -> None:
        for track in tracks:
            history = self._tracks.setdefault(track.track_id, {})
            for frame_index, centroid in track.history:
                history[frame_index] = centroid

    def summaries(self) -> List[TrackSummary]:
        summaries: List[TrackSummary] = []
        for track_id, history in self._tracks.items():
            if len(history) < 2:
                continue
            history_sorted = sorted(history.items(), key=lambda item: item[0])
            positions = [position for _, position in history_sorted]
            path_length_px = 0.0
            for start, end in zip(positions, positions[1:]):
                path_length_px += math.dist(start, end)
            path_length_m = path_length_px * self.config.meters_per_pixel
            total_frames = history_sorted[-1][0] - history_sorted[0][0] + 1
            duration_s = max(total_frames / self.fps, 1e-6)
            avg_speed = path_length_m / duration_s
            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    path_length_m=float(path_length_m),
                    avg_speed_m_per_s=float(avg_speed),
                    total_frames=int(total_frames),
                )
            )
        return summaries

    def to_dataframe(self) -> "pd.DataFrame":
        if pd is None:
            raise ImportError("pandas is required to export movement summaries as a DataFrame")
        data = [summary.as_dict() for summary in self.summaries()]
        return pd.DataFrame(data)

