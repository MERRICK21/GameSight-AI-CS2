"""Unit tests for IOUTracker and _iou helper."""

from unittest import TestCase

from gamesight.domain.models import Detection
from gamesight.tracking.tracker import IOUTracker, MultiObjectTracker, _iou


def _det(label: str, x1: float, y1: float, x2: float, y2: float, fi: int = 0, ts: float = 0.0) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_xyxy=(x1, y1, x2, y2), frame_index=fi, timestamp_sec=ts)


# -- IOU helper tests --------------------------------------------------------

class IOUHelperTests(TestCase):
    def test_perfect_overlap(self) -> None:
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_no_overlap(self) -> None:
        self.assertEqual(_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (5, 5, 15, 15)), 0.14285714285714285)

    def test_one_inside_another(self) -> None:
        self.assertAlmostEqual(_iou((0, 0, 100, 100), (20, 20, 80, 80)), 0.36)

    def test_zero_area_boxes(self) -> None:
        self.assertEqual(_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_edge_touching_no_overlap(self) -> None:
        self.assertEqual(_iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)


# -- IOUTracker tests --------------------------------------------------------

class IOUTrackerInterfaceTests(TestCase):
    def test_implements_multi_object_tracker(self) -> None:
        tracker = IOUTracker()
        self.assertIsInstance(tracker, MultiObjectTracker)

    def test_default_constructor_values(self) -> None:
        tracker = IOUTracker()
        self.assertEqual(tracker._iou_threshold, 0.3)
        self.assertEqual(tracker._max_lost, 30)

    def test_raises_on_invalid_iou_threshold(self) -> None:
        with self.assertRaises(ValueError):
            IOUTracker(iou_threshold=0.0)
        with self.assertRaises(ValueError):
            IOUTracker(iou_threshold=1.5)

    def test_raises_on_invalid_max_lost(self) -> None:
        with self.assertRaises(ValueError):
            IOUTracker(max_lost_frames=0)

    def test_update_returns_sequence(self) -> None:
        tracker = IOUTracker()
        result = tracker.update([])
        self.assertIsInstance(result, tuple)


class IOUTrackerTrackingTests(TestCase):
    def test_empty_detections_returns_empty(self) -> None:
        tracker = IOUTracker()
        tracks = tracker.update([])
        self.assertEqual(len(tracks), 0)

    def test_first_detection_creates_track(self) -> None:
        tracker = IOUTracker()
        tracks = tracker.update([_det("enemy", 100, 200, 180, 400)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, "track_0000")
        self.assertEqual(tracks[0].label, "enemy")
        self.assertEqual(len(tracks[0].detections), 1)

    def test_consecutive_detections_associate(self) -> None:
        tracker = IOUTracker(iou_threshold=0.3)

        # Frame 1
        t1 = tracker.update([_det("enemy", 100, 200, 180, 400, fi=1)])
        self.assertEqual(len(t1), 1)
        tid = t1[0].track_id

        # Frame 2: near-identical bbox 鈫?matches
        t2 = tracker.update([_det("enemy", 102, 198, 182, 402, fi=2)])
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].track_id, tid, "Track ID should persist")
        self.assertEqual(len(t2[0].detections), 2, "Should accumulate detections")

    def test_different_bbox_spawns_new_track(self) -> None:
        tracker = IOUTracker(iou_threshold=0.5)

        tracker.update([_det("enemy", 100, 200, 180, 400)])
        # Far-away bbox 鈫?no IOU match 鈫?new track
        tracks = tracker.update([_det("enemy", 500, 500, 600, 700)])

        self.assertEqual(len(tracks), 2)
        ids = {t.track_id for t in tracks}
        self.assertEqual(len(ids), 2)

    def test_multiple_detections_multiple_tracks(self) -> None:
        tracker = IOUTracker(iou_threshold=0.3)

        # Frame 1: two players
        tracker.update([
            _det("enemy", 100, 200, 180, 400),
            _det("teammate", 500, 200, 580, 400),
        ])

        # Frame 2: same two players with slight movement
        tracks = tracker.update([
            _det("enemy", 105, 205, 185, 405),
            _det("teammate", 495, 195, 585, 405),
        ])

        self.assertEqual(len(tracks), 2)
        for t in tracks:
            self.assertEqual(len(t.detections), 2)
        labels = {t.label for t in tracks}
        self.assertEqual(labels, {"enemy", "teammate"})

    def test_track_termination_after_lost(self) -> None:
        tracker = IOUTracker(iou_threshold=0.5, max_lost_frames=2)

        # Create a track
        tracker.update([_det("enemy", 100, 200, 180, 400)])

        # 2 empty frames 鈫?track pruned
        tracker.update([])
        tracks = tracker.update([])

        self.assertEqual(len(tracks), 0, "Track should be pruned after 2 lost frames")

    def test_track_survives_within_max_lost(self) -> None:
        tracker = IOUTracker(iou_threshold=0.5, max_lost_frames=3)

        tracker.update([_det("enemy", 100, 200, 180, 400)])

        # 2 empty frames 鈥?track still alive
        tracker.update([])
        tracks = tracker.update([])
        self.assertEqual(len(tracks), 1)

    def test_reset_clears_everything(self) -> None:
        tracker = IOUTracker()
        tracker.update([_det("enemy", 100, 200, 180, 400)])
        self.assertEqual(len(tracker.update([])), 1)

        tracker.reset()
        self.assertEqual(len(tracker._tracks), 0)
        self.assertEqual(tracker._next_id, 0)
        self.assertEqual(len(tracker.update([])), 0)

    def test_track_id_increments(self) -> None:
        tracker = IOUTracker(max_lost_frames=2)

        tracker.update([_det("enemy", 100, 200, 180, 400)])
        tracker.update([])  # lose track
        tracker.update([])
        tracker.update([])  # pruned

        # New detection after pruning 鈫?new ID
        tracks = tracker.update([_det("enemy", 200, 300, 280, 500)])
        self.assertEqual(tracks[0].track_id, "track_0001")

    def test_label_preserved_in_track(self) -> None:
        tracker = IOUTracker(max_lost_frames=2)

        tracker.update([_det("enemy", 100, 200, 180, 400)])
        tracks = tracker.update([_det("player", 102, 198, 182, 402)])

        # The track label is set when created; not overwritten by classifier
        self.assertEqual(tracks[0].label, "enemy")

    def test_detection_timestamp_preserved(self) -> None:
        tracker = IOUTracker()
        tracker.update([_det("enemy", 100, 200, 180, 400, fi=5, ts=3.5)])
        tracks = tracker.update([_det("enemy", 102, 198, 182, 402, fi=6, ts=4.0)])

        dets = tracks[0].detections
        self.assertEqual(dets[0].frame_index, 5)
        self.assertAlmostEqual(dets[0].timestamp_sec, 3.5)
        self.assertEqual(dets[1].frame_index, 6)
        self.assertAlmostEqual(dets[1].timestamp_sec, 4.0)
