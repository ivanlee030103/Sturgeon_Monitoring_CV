import math

from sturgeon_monitoring.config import MovementConfig
from sturgeon_monitoring.detection import BoundingBox
from sturgeon_monitoring.movement import MovementAnalyzer
from sturgeon_monitoring.tracking import Track


def test_movement_analyzer_computes_distance_and_speed():
    analyzer = MovementAnalyzer(MovementConfig(meters_per_pixel=0.01), fps=10)
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0)
    track = Track(
        track_id=1,
        bbox=bbox,
        centroid=(20.0, 5.0),
        frame_index=2,
        history=[(0, (0.0, 5.0)), (1, (10.0, 5.0)), (2, (20.0, 5.0))],
    )

    analyzer.update([track])
    summaries = analyzer.summaries()
    assert len(summaries) == 1

    summary = summaries[0]
    assert math.isclose(summary.path_length_m, 0.2, rel_tol=1e-5)
    # Path length 0.2m over 0.3s (3 frames at 10 fps)
    assert math.isclose(summary.avg_speed_m_per_s, 0.2 / 0.3, rel_tol=1e-5)
    assert summary.total_frames == 3

