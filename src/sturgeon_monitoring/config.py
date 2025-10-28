"""Configuration objects for the sturgeon monitoring pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration controlling YOLO inference."""

    model_path: str | Path = "yolo11n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: Optional[str] = None


@dataclass
class TrackerConfig:
    """Configuration for the centroid-based multi-object tracker."""

    max_distance: float = 80.0
    max_missed_frames: int = 30


@dataclass
class MovementConfig:
    """Parameters for computing movement statistics."""

    meters_per_pixel: float = 1.0
    smoothing_window: int = 5

