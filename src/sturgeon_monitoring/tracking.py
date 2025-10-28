"""Simple centroid-based multi-object tracking for sturgeon detection."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import math

from .config import TrackerConfig
from .detection import BoundingBox


@dataclass
class Track:
    """Represents a tracked sturgeon across frames."""

    track_id: int
    bbox: BoundingBox
    centroid: Tuple[float, float]
    frame_index: int
    history: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """Serialise the track for downstream reporting."""

        return {
            "track_id": self.track_id,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
                "confidence": self.bbox.confidence,
                "class_id": self.bbox.class_id,
            },
            "centroid": {"x": self.centroid[0], "y": self.centroid[1]},
            "frame_index": self.frame_index,
        }


class CentroidTracker:
    """A lightweight tracker matching detections based on centroid distance."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._next_id = 1
        self._tracks: "OrderedDict[int, TrackState]" = OrderedDict()

    def update(self, detections: Iterable[BoundingBox], frame_index: int) -> List[Track]:
        """Update active tracks based on new detections."""

        detections = list(detections)
        if not detections:
            self._increment_missed()
            return []

        input_centroids = [_centroid(det) for det in detections]

        if not self._tracks:
            return self._register_new_tracks(detections, input_centroids, frame_index)

        object_ids = list(self._tracks.keys())
        object_centroids = [state.centroid for state in self._tracks.values()]

        distance_matrix = _distance_matrix(object_centroids, input_centroids)
        if not distance_matrix:
            distance_matrix = [[] for _ in object_centroids]
        rows = sorted(range(len(distance_matrix)), key=lambda r: min(distance_matrix[r]) if distance_matrix[r] else math.inf)

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        updates: List[Track] = []

        for row in rows:
            if row >= len(object_centroids):
                continue
            if not distance_matrix[row]:
                continue
            col, distance = min(
                ((col_idx, distance_matrix[row][col_idx]) for col_idx in range(len(input_centroids)) if col_idx not in assigned_detections),
                key=lambda item: item[1],
                default=(None, math.inf),
            )
            if col is None or distance > self.config.max_distance:
                continue

            object_id = object_ids[row]
            state = self._tracks[object_id]
            centroid = input_centroids[col]
            bbox = detections[col]
            state.update(bbox, centroid, frame_index)
            updates.append(state.as_track())
            assigned_tracks.add(object_id)
            assigned_detections.add(col)

        self._increment_missed(exclude=assigned_tracks)

        # Register detections that could not be matched
        for idx, bbox in enumerate(detections):
            if idx in assigned_detections:
                continue
            centroid = input_centroids[idx]
            updates.extend(self._register(bbox, centroid, frame_index))

        return updates

    def active_tracks(self) -> List[Track]:
        """Return tracks that have not timed out."""

        return [state.as_track() for state in self._tracks.values() if not state.should_deregister]

    # Internal helpers -------------------------------------------------

    def _register_new_tracks(
        self, detections: List[BoundingBox], centroids: List[Tuple[float, float]], frame_index: int
    ) -> List[Track]:
        tracks: List[Track] = []
        for bbox, centroid in zip(detections, centroids):
            tracks.extend(self._register(bbox, centroid, frame_index))
        return tracks

    def _register(self, bbox: BoundingBox, centroid: Tuple[float, float], frame_index: int) -> List[Track]:
        state = TrackState(
            track_id=self._next_id,
            bbox=bbox,
            centroid=centroid,
            frame_index=frame_index,
            max_missed=self.config.max_missed_frames,
        )
        self._tracks[self._next_id] = state
        self._next_id += 1
        return [state.as_track()]

    def _increment_missed(self, exclude: Iterable[int] | None = None) -> None:
        exclude_set = set(exclude or [])
        to_delete = []
        for track_id, state in self._tracks.items():
            if track_id in exclude_set:
                continue
            state.mark_missed()
            if state.should_deregister:
                to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks.pop(track_id, None)


class TrackState:
    """Internal tracker state for a single target."""

    def __init__(
        self,
        track_id: int,
        bbox: BoundingBox,
        centroid: Tuple[float, float],
        frame_index: int,
        max_missed: int,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = centroid
        self.frame_index = frame_index
        self.max_missed = max_missed
        self.missed = 0
        self.history: List[Tuple[int, Tuple[float, float]]] = [(frame_index, centroid)]

    def update(self, bbox: BoundingBox, centroid: Tuple[float, float], frame_index: int) -> None:
        self.bbox = bbox
        self.centroid = centroid
        self.frame_index = frame_index
        self.history.append((frame_index, centroid))
        self.missed = 0

    def mark_missed(self) -> None:
        self.missed += 1

    @property
    def should_deregister(self) -> bool:
        return self.missed > self.max_missed

    def as_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            centroid=self.centroid,
            frame_index=self.frame_index,
            history=list(self.history),
        )


def _centroid(bbox: BoundingBox) -> Tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _distance_matrix(
    objects: List[Tuple[float, float]], detections: List[Tuple[float, float]]
) -> List[List[float]]:
    matrix: List[List[float]] = []
    for obj in objects:
        row = [math.dist(obj, det) for det in detections]
        matrix.append(row)
    return matrix

