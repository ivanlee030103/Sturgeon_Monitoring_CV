import numpy as np

from sturgeon_monitoring.detection import BoundingBox
from sturgeon_monitoring.heatmap import (
    HeatmapIntervalSaver,
    IntervalHeatmapConfig,
    LiveHeatmap,
    LiveHeatmapConfig,
)
from sturgeon_monitoring.tracking import Track


def make_track(track_id: int, x: float, y: float, frame_index: int) -> Track:
    bbox = BoundingBox(x1=x - 5, y1=y - 5, x2=x + 5, y2=y + 5, confidence=0.9, class_id=0)
    history = [(frame_index, (x, y))]
    return Track(track_id=track_id, bbox=bbox, centroid=(x, y), frame_index=frame_index, history=history)


def test_live_heatmap_render_shape():
    heatmap = LiveHeatmap(width=200, height=100, config=LiveHeatmapConfig(decay=1.0, blur_kernel_size=0))
    track = make_track(1, 50.0, 25.0, frame_index=0)
    heatmap.update([track])
    rendered = heatmap.render()
    assert rendered.shape == (100, 200, 3)
    assert np.any(rendered)


def test_interval_saver_creates_plots(tmp_path):
    config = IntervalHeatmapConfig(interval_seconds=0.5)
    saver = HeatmapIntervalSaver(
        width=320,
        height=180,
        fps=10.0,
        video_stem="sample",
        output_dir=tmp_path,
        config=config,
    )

    for frame_index in range(10):
        track = make_track(1, 20.0 + frame_index, 40.0, frame_index)
        saver.observe([track], frame_index)

    saver.finalize()
    saved = saver.saved_windows
    assert saved
    for heatmap_path, scatter_path in saved:
        assert heatmap_path.exists()
        assert scatter_path.exists()
