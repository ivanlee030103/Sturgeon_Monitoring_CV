"""Command line pipeline for monitoring sturgeon movement with YOLOv11."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import cv2
from .config import DetectorConfig, MovementConfig, TrackerConfig
from .detection import BoundingBox, SturgeonDetector
from .movement import MovementAnalyzer
from .tracking import CentroidTracker
from .visualization import draw_tracks


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="Path to the YOLOv11 weights file (downloaded via Ultralytics).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for outputs.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for non-maximum suppression.")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run inference on.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[],
        help="Restrict detection to specific class IDs (empty for all classes).",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=1.0,
        help="Scaling factor to convert pixels to meters for path length calculations.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override the FPS reported by the video file.")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Persist an annotated video with tracking overlays in the output directory.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated frames in a window while processing (press q to quit).",
    )
    return parser.parse_args(args=args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_config = DetectorConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
    )
    tracker_config = TrackerConfig()
    movement_config = MovementConfig(meters_per_pixel=args.meters_per_pixel)

    detector = SturgeonDetector(detector_config)
    tracker = CentroidTracker(tracker_config)

    video_path = args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = args.fps or capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    movement = MovementAnalyzer(movement_config, fps=fps)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = output_dir / f"{video_path.stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

    frame_index = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            detections = detector.detect(frame)
            detections = _filter_classes(detections, args.classes)
            tracks = tracker.update(detections, frame_index)
            movement.update(tracks)

            draw_tracks(frame, tracks)

            if writer is not None:
                writer.write(frame)

            if args.display:
                cv2.imshow("Sturgeon Monitoring", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    _save_results(movement, output_dir)


def _filter_classes(detections: List[BoundingBox], classes: Iterable[int]) -> List[BoundingBox]:
    if not classes:
        return detections
    allowed = set(classes)
    return [det for det in detections if det.class_id in allowed]


def _save_results(movement: MovementAnalyzer, output_dir: Path) -> None:
    df = movement.to_dataframe()
    csv_path = output_dir / "movement_summary.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_dir / "movement_summary.json"
    json_data = df.to_dict(orient="records")
    json_path.write_text(json.dumps(json_data, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

