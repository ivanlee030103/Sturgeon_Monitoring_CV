"""YOLOv11 detector wrapper for sturgeon monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional import for tests without numpy
    import numpy as np
except Exception:  # pragma: no cover - handled dynamically
    np = None

from .config import DetectorConfig

try:  # pragma: no cover - exercised in integration environments only
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - handled during runtime
    YOLO = None
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - only defined when import succeeds
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class BoundingBox:
    """Representation of a detection bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    def as_xyxy(self) -> "np.ndarray | tuple[float, float, float, float]":
        if np is None:
            return (self.x1, self.y1, self.x2, self.y2)
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=float)


class SturgeonDetector:
    """Runs YOLOv11 inference on frames to detect sturgeon."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        if YOLO is None:
            raise ImportError(
                "Failed to import ultralytics. Ensure the 'ultralytics' package is installed"
            ) from _IMPORT_ERROR
        self.model = YOLO(str(self.config.model_path))
        if self.config.device:
            self.model.to(self.config.device)

    def detect(self, frame: "np.ndarray") -> List[BoundingBox]:
        """Run inference on a BGR frame and return detected bounding boxes."""

        if np is None:
            raise ImportError("numpy is required for running detections")

        if frame is None:
            raise ValueError("Frame cannot be None")

        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        detections: List[BoundingBox] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
            classes = (
                boxes.cls.cpu().numpy().astype(int)
                if boxes.cls is not None
                else np.full(len(xyxy), -1, dtype=int)
            )
            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
                detections.append(
                    BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(conf),
                        class_id=int(class_id),
                    )
                )
        return detections

    @staticmethod
    def filter_by_class(
        detections: Iterable[BoundingBox], allowed_classes: Iterable[int]
    ) -> List[BoundingBox]:
        """Filter detections to only keep specific class identifiers."""

        allowed = set(allowed_classes)
        if not allowed:
            return list(detections)
        return [det for det in detections if det.class_id in allowed]

