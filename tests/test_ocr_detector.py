import unittest

import numpy as np

from gamesight.domain.models import EventType, HudState
from gamesight.events.ocr_detector import OCRRoundDetector
from gamesight.perception.ocr import _parse_score_value


class _ScoreReader:
    available = True

    def __init__(self, scores):
        self.scores = iter(scores)

    def read(self, crop, frame_index, timestamp_sec):
        ct, t = next(self.scores)
        return {"ct_score": ct, "t_score": t, "round_number": ct + t + 1}


def _state(ts, timer):
    return HudState(
        frame_index=int(ts * 30), timestamp_sec=ts, confidence=1.0,
        values={"round_info.timer_visible": timer},
    )


class ScoreParsingTests(unittest.TestCase):
    def test_parses_normal_and_two_digit_scores(self):
        self.assertEqual(_parse_score_value(["0"]), 0)
        self.assertEqual(_parse_score_value(["12"]), 12)

    def test_removes_vertical_separator_artifact(self):
        self.assertEqual(_parse_score_value(["71"]), 1)
        self.assertEqual(_parse_score_value(["74"]), 4)


class OCRRoundDetectorTests(unittest.TestCase):
    def test_timer_updates_do_not_run_ocr_between_scoreboards(self):
        detector = OCRRoundDetector()
        reader = _ScoreReader([(0, 0)])
        detector._reader = reader
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        detector.update(_state(0, True), image, read_score=True)
        started = detector.update(_state(.5, True), read_score=False)
        self.assertEqual(
            [event.event_type for event in started], [EventType.ROUND_START],
        )
        self.assertEqual(detector.update(_state(20, True), read_score=False), ())
        with self.assertRaises(StopIteration):
            detector.update(_state(21, False), image, read_score=True)

    def test_waits_for_timer_before_starting_next_round(self):
        detector = OCRRoundDetector()
        detector._reader = _ScoreReader([(0, 0), (0, 0), (0, 1), (0, 1), (0, 1), (0, 1)])
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        self.assertEqual(detector.update(_state(0, True), image), ())
        first = detector.update(_state(1, True), image)
        unconfirmed = detector.update(_state(49, False), image)
        score_change = detector.update(_state(50, False), image)
        self.assertEqual(detector.update(_state(55, True), image), ())
        next_start = detector.update(_state(56, True), image)

        self.assertEqual([e.event_type for e in first], [EventType.ROUND_START])
        self.assertEqual(unconfirmed, ())
        self.assertEqual([e.event_type for e in score_change], [EventType.ROUND_END])
        self.assertEqual([e.event_type for e in next_start], [EventType.ROUND_START])

    def test_final_score_does_not_create_phantom_round(self):
        detector = OCRRoundDetector()
        detector._reader = _ScoreReader([(12, 5), (12, 5), (13, 5), (13, 5)])
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detector.update(_state(1100, True), image)
        detector.update(_state(1101, True), image)
        detector.update(_state(1159, False), image)
        final = detector.update(_state(1160, False), image)
        self.assertEqual([e.event_type for e in final], [EventType.ROUND_END])

    def test_ignores_score_jumps_and_single_frame_noise(self):
        detector = OCRRoundDetector()
        detector._reader = _ScoreReader([(0, 0), (0, 0), (5, 0), (0, 1), (0, 0)])
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detector.update(_state(0, True), image)
        detector.update(_state(1, True), image)
        self.assertEqual(detector.update(_state(10, True), image), ())
        self.assertEqual(detector.update(_state(20, True), image), ())
        self.assertEqual(detector.update(_state(30, True), image), ())


if __name__ == "__main__":
    unittest.main()
