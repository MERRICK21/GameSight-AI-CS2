"""Unit tests for PlayerClassifier."""

from unittest import TestCase

import numpy as np

from gamesight.domain.models import Detection
from gamesight.perception.classifier import PlayerClassifier


def _det(label: str = "player", x1: int = 100, y1: int = 200, x2: int = 180, y2: int = 400) -> Detection:
    return Detection(
        label=label, confidence=0.85,
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        frame_index=0, timestamp_sec=0.0,
    )


def _red_pixels(h: int, w: int) -> np.ndarray:
    """Pure red BGR image: (0, 0, 255)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)
    return img


def _blue_pixels(h: int, w: int) -> np.ndarray:
    """Pure blue BGR image: (255, 0, 0)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)
    return img


# -- PlayerClassifier tests --------------------------------------------------


class PlayerClassifierTests(TestCase):
    def setUp(self) -> None:
        self.classifier = PlayerClassifier()

    def test_red_crop_classified_as_enemy(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[200:400, 100:180] = (0, 0, 255)  # red in bbox
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "enemy")

    def test_blue_crop_classified_as_teammate(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[200:400, 100:180] = (255, 0, 0)  # blue in bbox
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "teammate")

    def test_ambiguous_crop_stays_player(self) -> None:
        """Equal red and blue should not classify."""
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        half = 100
        frame[200:200 + half, 100:180] = (0, 0, 255)
        frame[200 + half:400, 100:180] = (255, 0, 0)
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "player")

    def test_empty_crop_keeps_label(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "player")

    def test_multiple_detections_mixed(self) -> None:
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Player 1: red region
        frame[100:300, 50:150] = (0, 0, 255)
        # Player 2: blue region
        frame[100:300, 300:400] = (255, 0, 0)

        detections = [
            Detection(label="player", confidence=0.8, bbox_xyxy=(50, 100, 150, 300), frame_index=0, timestamp_sec=0.0),
            Detection(label="player", confidence=0.7, bbox_xyxy=(300, 100, 400, 300), frame_index=0, timestamp_sec=0.0),
        ]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "enemy")
        self.assertEqual(result[1].label, "teammate")

    def test_already_classified_preserved(self) -> None:
        """If a detection is already 'enemy', the classifier may override it."""
        frame = _blue_pixels(500, 500)
        detections = [Detection(label="enemy", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        # Blue crop dominates 鈫?should be teammate (classifier overrides)
        self.assertEqual(result[0].label, "teammate")

    def test_out_of_bounds_bbox_clamped(self) -> None:
        frame = _red_pixels(500, 500)
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(-10, -10, 600, 600), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "enemy")

    def test_empty_detections_returns_empty(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        result = self.classifier.classify(frame, [])
        self.assertEqual(result, [])

    def test_near_red_color_classified_as_enemy(self) -> None:
        """Slightly off-red (orange tint) still within range."""
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[200:400, 100:180] = (20, 40, 220)  # orange-red, within range
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "enemy")

    def test_custom_color_ranges(self) -> None:
        classifier = PlayerClassifier(
            red_low=(0, 0, 200),
            red_high=(50, 50, 255),
            blue_low=(200, 0, 0),
            blue_high=(255, 100, 50),
        )
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[200:400, 100:180] = (30, 30, 240)
        detections = [Detection(label="player", confidence=0.85, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0)]
        result = classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "enemy")

    def test_confidence_and_metadata_preserved(self) -> None:
        frame = _red_pixels(500, 500)
        detections = [Detection(label="player", confidence=0.88, bbox_xyxy=(100, 200, 180, 400), frame_index=42, timestamp_sec=3.14)]
        result = self.classifier.classify(frame, detections)
        self.assertEqual(result[0].label, "enemy")
        self.assertAlmostEqual(result[0].confidence, 0.88)
        self.assertEqual(result[0].frame_index, 42)
        self.assertAlmostEqual(result[0].timestamp_sec, 3.14)
