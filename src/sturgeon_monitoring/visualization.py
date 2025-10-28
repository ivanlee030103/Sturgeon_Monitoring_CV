"""Visualization helpers for sturgeon monitoring."""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2

from .tracking import Track


_COLOR = (0, 255, 0)
_TEXT_COLOR = (255, 255, 255)


def draw_tracks(frame, tracks: Iterable[Track]) -> None:
    """Draw bounding boxes and identifiers on the provided frame."""

    for track in tracks:
        bbox = track.bbox
        top_left = (int(bbox.x1), int(bbox.y1))
        bottom_right = (int(bbox.x2), int(bbox.y2))
        cv2.rectangle(frame, top_left, bottom_right, _COLOR, 2)
        label = f"ID {track.track_id}"
        cv2.putText(frame, label, _text_position(top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_COLOR, 2)


def _text_position(top_left: Tuple[int, int]) -> Tuple[int, int]:
    return top_left[0], max(top_left[1] - 10, 10)

