"""Sturgeon monitoring package built around YOLOv11 detection."""

from .config import DetectorConfig, MovementConfig, TrackerConfig
from .detection import BoundingBox, SturgeonDetector
from .heatmap import HeatmapIntervalSaver, IntervalHeatmapConfig, LiveHeatmap, LiveHeatmapConfig
from .movement import MovementAnalyzer, TrackSummary
from .tracking import CentroidTracker, Track

__all__ = [
    "DetectorConfig",
    "MovementConfig",
    "TrackerConfig",
    "BoundingBox",
    "SturgeonDetector",
    "MovementAnalyzer",
    "TrackSummary",
    "CentroidTracker",
    "Track",
    "LiveHeatmap",
    "LiveHeatmapConfig",
    "HeatmapIntervalSaver",
    "IntervalHeatmapConfig",
]

