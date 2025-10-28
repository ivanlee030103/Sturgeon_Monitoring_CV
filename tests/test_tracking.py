from sturgeon_monitoring.config import TrackerConfig
from sturgeon_monitoring.detection import BoundingBox
from sturgeon_monitoring.tracking import CentroidTracker


def make_box(x1, y1, width=20, height=10, confidence=0.9, class_id=0):
    return BoundingBox(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height, confidence=confidence, class_id=class_id)


def test_tracker_assigns_consistent_ids():
    tracker = CentroidTracker(TrackerConfig(max_distance=50.0, max_missed_frames=2))

    detections_frame0 = [make_box(10, 10), make_box(100, 20)]
    tracks_frame0 = tracker.update(detections_frame0, frame_index=0)
    assert len(tracks_frame0) == 2
    ids_frame0 = {track.track_id for track in tracks_frame0}

    detections_frame1 = [make_box(15, 12), make_box(105, 18)]
    tracks_frame1 = tracker.update(detections_frame1, frame_index=1)
    assert {track.track_id for track in tracks_frame1} == ids_frame0

    for track in tracks_frame1:
        assert len(track.history) >= 2


def test_tracker_handles_missing_detections():
    tracker = CentroidTracker(TrackerConfig(max_distance=50.0, max_missed_frames=1))

    tracks_frame0 = tracker.update([make_box(10, 10)], frame_index=0)
    track_id = tracks_frame0[0].track_id

    # No detections next frame; tracker should mark as missed but keep track alive until threshold exceeded
    assert tracker.update([], frame_index=1) == []
    assert any(t.track_id == track_id for t in tracker.active_tracks())

    # Another empty frame should deregister the track
    tracker.update([], frame_index=2)
    assert all(t.track_id != track_id for t in tracker.active_tracks())

