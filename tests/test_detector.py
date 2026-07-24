"""Unit tests for YOLODetector."""

from dataclasses import dataclass
from unittest import TestCase

import numpy as np

from gamesight.domain.models import Detection
from gamesight.perception.detector import ObjectDetector, YOLODetector


# -- mock helpers -------------------------------------------------------------


@dataclass
class _MockBox:
    """Minimal mock for a YOLO Results.boxes entry."""
    cls: object
    conf: object
    xyxy: object


class _MockResult:
    """Minimal mock for a single ultralytics Results object."""
    def __init__(self, boxes: list[_MockBox] | None = None) -> None:
        self.boxes = boxes


def _mock_model(*result_boxes: list[_MockBox]) -> object:
    """Return a callable that behaves like a YOLO model, returning fixed detections."""

    class _MockYOLO:
        def __call__(self, image, verbose=False):
            return [_MockResult(boxes=result_boxes)]

    return _MockYOLO()


def _box(cls: int, conf: float, xyxy: list[float]) -> _MockBox:
    return _MockBox(
        cls=np.array([cls]),
        conf=np.array([conf]),
        xyxy=np.array([xyxy]),
    )


# -- YOLODetector tests ------------------------------------------------------


class YOLODetectorInterfaceTests(TestCase):
    """Contract and construction tests."""

    def test_implements_object_detector(self) -> None:
        detector = YOLODetector(model=_mock_model())
        self.assertIsInstance(detector, ObjectDetector)

    def test_detect_returns_sequence(self) -> None:
        detector = YOLODetector(model=_mock_model())
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, 0, 0.0)
        self.assertIsInstance(result, tuple)

    def test_non_numpy_frame_returns_empty(self) -> None:
        detector = YOLODetector(model=_mock_model())
        result = detector.detect("not an array", 0, 0.0)
        self.assertEqual(result, ())


class YOLODetectorDetectionTests(TestCase):
    """Tests for detection filtering logic."""

    def test_person_class_detected(self) -> None:
        model = _mock_model(
            _box(cls=0, conf=0.9, xyxy=[100, 200, 300, 500]),
        )
        detector = YOLODetector(model=model, confidence_threshold=0.5)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 5, 1.5)
        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertEqual(d.label, "player")
        self.assertAlmostEqual(d.confidence, 0.9)
        self.assertEqual(d.bbox_xyxy, (100.0, 200.0, 300.0, 500.0))
        self.assertEqual(d.frame_index, 5)
        self.assertAlmostEqual(d.timestamp_sec, 1.5)

    def test_multiple_persons_detected(self) -> None:
        model = _mock_model(
            _box(cls=0, conf=0.8, xyxy=[10, 20, 100, 200]),
            _box(cls=0, conf=0.7, xyxy=[300, 400, 500, 600]),
        )
        detector = YOLODetector(model=model, confidence_threshold=0.5)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].label, "player")
        self.assertEqual(detections[1].label, "player")

    def test_non_person_class_filtered_out(self) -> None:
        """A car (class 2) should be ignored."""
        model = _mock_model(
            _box(cls=0, conf=0.9, xyxy=[10, 10, 50, 50]),
            _box(cls=2, conf=0.9, xyxy=[100, 100, 200, 200]),  # car
        )
        detector = YOLODetector(model=model, confidence_threshold=0.5)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "player")

    def test_below_confidence_threshold_filtered(self) -> None:
        model = _mock_model(
            _box(cls=0, conf=0.9, xyxy=[10, 10, 50, 50]),
            _box(cls=0, conf=0.2, xyxy=[100, 100, 200, 200]),
        )
        detector = YOLODetector(model=model, confidence_threshold=0.5)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(detections[0].confidence, 0.5)

    def test_no_boxes_returns_empty(self) -> None:
        result_no_boxes = _mock_model()
        detector = YOLODetector(model=result_no_boxes)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(detections, ())

    def test_all_below_threshold_returns_empty(self) -> None:
        model = _mock_model(
            _box(cls=0, conf=0.1, xyxy=[10, 10, 50, 50]),
            _box(cls=0, conf=0.2, xyxy=[100, 100, 200, 200]),
        )
        detector = YOLODetector(model=model, confidence_threshold=0.5)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(detections, ())

    def test_custom_person_class_id(self) -> None:
        """Person class ID 0 is default; test with a fake ID."""
        model = _mock_model(
            _box(cls=0, conf=0.9, xyxy=[10, 10, 50, 50]),
            _box(cls=5, conf=0.9, xyxy=[100, 100, 200, 200]),
        )
        detector = YOLODetector(model=model, confidence_threshold=0.5, person_class_id=5)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(len(detections), 1)
        # Should be the class-5 box, not class-0
        self.assertAlmostEqual(detections[0].bbox_xyxy[0], 100.0)

    def test_detection_bbox_is_float_tuple(self) -> None:
        model = _mock_model(
            _box(cls=0, conf=0.9, xyxy=[10.5, 20.5, 100.5, 200.5]),
        )
        detector = YOLODetector(model=model)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        bbox = detections[0].bbox_xyxy
        self.assertIsInstance(bbox, tuple)
        self.assertTrue(all(isinstance(v, float) for v in bbox))

    def test_grayscale_frame_still_processed(self) -> None:
        """Greyscale frames should not crash (YOLO handles them)."""
        model = _mock_model(
            _box(cls=0, conf=0.9, xyxy=[10, 10, 50, 50]),
        )
        detector = YOLODetector(model=model)
        frame = np.zeros((640, 640), dtype=np.uint8)

        detections = detector.detect(frame, 0, 0.0)
        self.assertEqual(len(detections), 1)
