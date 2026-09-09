"""Tests for evidence screenshots — models and extractor."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from gamesight.domain.models import (
    AnalysisResult, EventType, Evidence, GameEvent, RoundAnalysis,
    VideoInput, VideoMetadata,
)
from gamesight.evidence.models import EvidenceClip, EvidenceImage
from gamesight.evidence.extractor import (
    EvidenceClipExtractor,
    OpenCVScreenshotExtractor,
    build_round_keyframe_events,
)


class _MockCV2:
    """Mock OpenCV — returns valid frames for all seek positions."""

    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 1

    def __init__(self) -> None:
        self._opened = True

    def VideoCapture(self, path: str) -> _MockCV2:
        return _MockCV2()

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        if prop_id == self.CAP_PROP_FRAME_WIDTH:
            return 640.0
        if prop_id == self.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        return 0.0

    def set(self, prop_id: int, value: float) -> None:
        pass

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass

    @staticmethod
    def imwrite(path: str, img: np.ndarray) -> bool:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mock_png")
        return True


class _ClosedMockCV2:
    """Mock where VideoCapture.isOpened() returns False."""

    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 1

    def VideoCapture(self, path: str) -> _ClosedMockCV2:
        return _ClosedMockCV2()

    def isOpened(self) -> bool:
        return False

    def get(self, prop_id: int) -> float:
        return 0.0

    def set(self, prop_id: int, value: float) -> None:
        pass

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        pass

    @staticmethod
    def imwrite(path: str, img: np.ndarray) -> bool:
        return False


class EvidenceImageModelTests(TestCase):
    def test_construction(self) -> None:
        img = EvidenceImage(
            image_id="ev_001", event_id="kill_001", frame_index=150,
            timestamp_sec=5.0, image_path="/tmp/test.png",
            source="KillEventDetector", width=1920, height=1080,
        )
        self.assertEqual(img.image_id, "ev_001")
        self.assertEqual(img.width, 1920)

    def test_path_property(self) -> None:
        img = EvidenceImage(
            image_id="e1", event_id="e1", frame_index=1,
            timestamp_sec=1.0, image_path="/a/b.png", source="t",
        )
        self.assertEqual(img.path, Path("/a/b.png"))

    def test_exists_returns_bool(self) -> None:
        img = EvidenceImage(
            image_id="e2", event_id="e2", frame_index=1,
            timestamp_sec=1.0, image_path="/nonexistent/xyz.png", source="t",
        )
        self.assertFalse(img.exists())


class EvidenceClipTests(TestCase):
    def test_model_path_and_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(b"video")
            clip = EvidenceClip(
                clip_id="clip_e1", event_id="e1", trigger_sec=1.0,
                start_sec=.5, end_sec=1.5, video_path=str(path), source="test",
            )
            self.assertEqual(clip.path, path)
            self.assertTrue(clip.exists())

    def test_extracts_browser_compatible_short_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                (320, 180),
            )
            self.assertTrue(writer.isOpened())
            for index in range(60):
                frame = np.full((180, 320, 3), index * 4, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            event = GameEvent(
                event_id="engagement_r1_01",
                event_type=EventType.ENGAGEMENT_CANDIDATE,
                start_sec=1.0,
                confidence=.8,
                evidence=[Evidence(
                    timestamp_sec=1.0, frame_index=30, source="test",
                )],
                attributes={"clip_trigger_sec": 1.0},
            )
            clips = EvidenceClipExtractor(
                before_sec=.5, after_sec=.5, output_fps=15, max_width=160,
            ).extract(source, [event], root / "clips")
            self.assertEqual(len(clips), 1)
            self.assertTrue(clips[0].exists())
            self.assertEqual((clips[0].width, clips[0].height), (160, 90))
            capture = cv2.VideoCapture(str(clips[0].path))
            self.assertTrue(capture.isOpened())
            self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 14)
            capture.release()


class ScreenshotExtractorTests(TestCase):
    def setUp(self) -> None:
        self.mock_cv2 = _MockCV2()
        self.extractor = OpenCVScreenshotExtractor(cv2_module=self.mock_cv2, max_screenshots=10)

    def _event(self, ts: float, frame: int) -> GameEvent:
        return GameEvent(
            event_id=f"kill_{frame}",
            event_type=EventType.PLAYER_KILL,
            start_sec=ts,
            confidence=0.9,
            evidence=[Evidence(timestamp_sec=ts, frame_index=frame, source="test")],
        )

    def test_extract_returns_list(self) -> None:
        events = [self._event(5.0, 150)]
        with tempfile.TemporaryDirectory() as tmp:
            images = self.extractor.extract("fake.mp4", events, Path(tmp))
            self.assertIsInstance(images, list)

    def test_extract_creates_images(self) -> None:
        events = [self._event(5.0, 150), self._event(10.0, 300)]
        with tempfile.TemporaryDirectory() as tmp:
            images = self.extractor.extract("fake.mp4", events, Path(tmp))
            self.assertEqual(len(images), 2)
            for img in images:
                self.assertTrue(img.exists())

    def test_extract_respects_max_screenshots(self) -> None:
        limited = OpenCVScreenshotExtractor(cv2_module=_MockCV2(), max_screenshots=1)
        events = [self._event(1.0, 30), self._event(2.0, 60), self._event(3.0, 90)]
        with tempfile.TemporaryDirectory() as tmp:
            images = limited.extract("fake.mp4", events, Path(tmp))
            self.assertEqual(len(images), 1)

    def test_extract_image_has_metadata(self) -> None:
        events = [self._event(5.0, 150)]
        with tempfile.TemporaryDirectory() as tmp:
            images = self.extractor.extract("fake.mp4", events, Path(tmp))
            img = images[0]
            self.assertEqual(img.event_id, "kill_150")
            self.assertEqual(img.frame_index, 150)
            self.assertEqual(img.timestamp_sec, 5.0)
            self.assertEqual(img.width, 640)
            self.assertEqual(img.height, 480)

    def test_extract_no_evidence_frame_fallback(self) -> None:
        ev = GameEvent(
            event_id="no_frame",
            event_type=EventType.PLAYER_KILL,
            start_sec=5.0,
            confidence=0.9,
            evidence=[Evidence(timestamp_sec=5.0, frame_index=None, source="test")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            images = self.extractor.extract("fake.mp4", [ev], Path(tmp))
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].frame_index, 150)

    def test_extract_empty_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = self.extractor.extract("fake.mp4", [], Path(tmp))
            self.assertEqual(images, [])

    def test_extract_unopenable_video(self) -> None:
        extractor = OpenCVScreenshotExtractor(cv2_module=_ClosedMockCV2())
        events = [self._event(1.0, 30)]
        with tempfile.TemporaryDirectory() as tmp:
            images = extractor.extract("bad.mp4", events, Path(tmp))
            self.assertEqual(images, [])


class RoundKeyframeEventTests(TestCase):

    def _analysis(self) -> AnalysisResult:
        return AnalysisResult(
            video=VideoInput(video_id="v", path=Path("v.mp4")),
            metadata=VideoMetadata(
                duration_sec=120.0, fps=30.0, width=1920, height=1080,
            ),
            rounds=[
                RoundAnalysis(round_id="round_001", start_sec=0.0, end_sec=60.0),
                RoundAnalysis(round_id="round_002", start_sec=60.0, end_sec=120.0),
            ],
        )

    def test_builds_two_interior_frames_per_round(self) -> None:
        events = build_round_keyframe_events(self._analysis())
        self.assertEqual(len(events), 4)
        self.assertEqual([round(e.start_sec, 1) for e in events], [20.0, 40.0, 80.0, 100.0])
        self.assertTrue(all(e.event_type == EventType.KEYFRAME for e in events))
        self.assertEqual(events[0].evidence[0].frame_index, 600)
        self.assertEqual(events[2].attributes["round_id"], "round_002")

    def test_respects_limit(self) -> None:
        events = build_round_keyframe_events(
            self._analysis(), samples_per_round=3, max_events=3,
        )
        self.assertEqual(len(events), 3)

    def test_uses_video_end_for_truncated_round(self) -> None:
        analysis = self._analysis()
        analysis.rounds[-1].end_sec = None
        events = build_round_keyframe_events(analysis, samples_per_round=1)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1].start_sec, 90.0)
