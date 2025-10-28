# Sturgeon Monitoring with YOLOv11

This project provides a computer vision pipeline that detects and tracks sturgeon in underwater footage using the [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/) model. The tracker estimates the movement of each detected fish and summarises their trajectories.

## Features

- YOLOv11 inference via the `ultralytics` Python package.
- Lightweight centroid-based multi-object tracker to maintain fish identities across frames.
- Movement analytics including path length and average speed per tracked sturgeon.
- Optional saving of annotated videos and CSV/JSON reports for downstream analysis.

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or train a YOLOv11 weights file (e.g. `yolo11n.pt`) using the Ultralytics CLI.

## Usage

Run the end-to-end pipeline on a recorded video:

```bash
python -m sturgeon_monitoring.pipeline \
    path/to/sturgeon_video.mp4 \
    --model path/to/yolo11n.pt \
    --output-dir outputs \
    --save-video
```

Key options:

- `--classes`: restrict detections to YOLO class IDs that correspond to sturgeon in your dataset.
- `--meters-per-pixel`: calibrate distance calculations with a known scale from the camera setup.
- `--fps`: override the frames-per-second value if it is missing from the video metadata.
- `--display`: preview the annotated feed while processing (press `q` to exit).

The pipeline writes:

- `outputs/<video_name>_annotated.mp4` (when `--save-video` is used) containing bounding boxes and IDs.
- `outputs/movement_summary.csv` and `outputs/movement_summary.json` containing per-track metrics.

## Development

Run the test suite:

```bash
pytest
```

The package code lives in `src/sturgeon_monitoring/` and is organised into detection, tracking, movement analysis, and visualization modules.

