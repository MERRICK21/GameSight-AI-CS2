"""Integration tests for VideoAnalysisPipeline end-to-end path."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest import TestCase

import numpy as np

from gamesight.domain.models import Detection, VideoInput, VideoMetadata
from gamesight.ingestion.video_reader import VideoFrame, VideoReader
from gamesight.perception.detector import ObjectDetector
from gamesight.orchestration.pipeline import VideoAnalysisPipeline
from gamesight.perception.extractors import (
    CrosshairExtractor,
    HPBarExtractor,
    KillFeedExtractor,
    MoneyExtractor,
    RoundInfoExtractor,
)
from gamesight.perception.hud_parser import CS2HudParser
from gamesight.perception.hud_profiles import CS2_STANDARD_16X9


class _MockReader(VideoReader):
    """A test-double VideoReader that returns fixed metadata and synthetic frames."""

    def __init__(
        self,
        metadata: VideoMetadata | None = None,
        frames: list[VideoFrame] | None = None,
    ) -> None:
        self._metadata = metadata or VideoMetadata(
            fps=60.0, width=1920, height=1080, duration_sec=10.0, codec="mock"
        )
        self._frames = frames or []

    def inspect(self, video: VideoInput) -> VideoMetadata:
        return self._metadata

    def frames(self, video: VideoInput, sample_fps: float) -> Iterator[VideoFrame]:
        yield from self._frames


def _blank_frame(index: int, ts: float) -> VideoFrame:
    return VideoFrame(
        frame_index=index,
        timestamp_sec=ts,
        image=np.zeros((1080, 1920, 3), dtype=np.uint8),
    )


def _crosshair_frame(index: int, ts: float) -> VideoFrame:
    """Frame with a visible crosshair at centre."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cx, cy = 960, 540
    img[cy - 2 : cy + 2, cx - 20 : cx + 20, :] = [50, 220, 50]
    img[cy - 15 : cy + 15, cx - 2 : cx + 2, :] = [50, 220, 50]
    return VideoFrame(frame_index=index, timestamp_sec=ts, image=img)


def _hp_frame(index: int, ts: float) -> VideoFrame:
    """Frame with full green HP bar and blue armour."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    green = (50, 220, 50)
    blue = (160, 100, 20)
    # HP bar zone: player_status at [556..1362, 955..1073]
    img[1020:1073, 556:1362, :] = green
    # Armour zone
    img[955:984, 556:1362, :] = blue
    return VideoFrame(frame_index=index, timestamp_sec=ts, image=img)


def _kill_feed_frame(index: int, ts: float) -> VideoFrame:
    """Frame with white kill-feed text."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    white = (255, 255, 255)
    # kill_feed region: [1488..1900, 5..264]
    img[20:30, 1500:1560, :] = white
    img[50:60, 1500:1580, :] = white
    img[80:90, 1500:1600, :] = white
    return VideoFrame(frame_index=index, timestamp_sec=ts, image=img)


def _round_info_frame(index: int, ts: float) -> VideoFrame:
    """Frame with visible round timer."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    white = (255, 255, 255)
    # round_info region: [652..1266, 5..96]
    img[20:40, 700:900, :] = white
    return VideoFrame(frame_index=index, timestamp_sec=ts, image=img)


def _money_frame(index: int, ts: float) -> VideoFrame:
    """Frame with visible money text."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    yellow = (20, 220, 220)
    # money region: [9..239, 1015..1074]
    img[1025:1040, 30:100, :] = yellow
    return VideoFrame(frame_index=index, timestamp_sec=ts, image=img)


# -- pipeline tests ----------------------------------------------------------

class VideoAnalysisPipelineTests(TestCase):
    """End-to-end tests wiring mock reader through real CS2HudParser."""

    def setUp(self) -> None:
        self.video = VideoInput(video_id="test-001", path=Path("test.mp4"))
        self.metadata = VideoMetadata(
            fps=60.0, width=1920, height=1080, duration_sec=10.0, codec="mock"
        )
        self.parser = CS2HudParser(
            CS2_STANDARD_16X9,
            {
                "crosshair": CrosshairExtractor(),
                "player_status": HPBarExtractor(),
                "kill_feed": KillFeedExtractor(),
                "money": MoneyExtractor(),
                "round_info": RoundInfoExtractor(),
            },
        )

    def test_empty_video_produces_zero_frames_summary(self) -> None:
        reader = _MockReader(metadata=self.metadata, frames=[])
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        self.assertEqual(result.video.video_id, "test-001")
        self.assertEqual(result.metadata.fps, 60.0)
        self.assertEqual(result.rounds, [])
        self.assertIn("[pipeline] 0 frames processed", result.warnings)

    def test_single_blank_frame_all_hud_false(self) -> None:
        frames = [_blank_frame(0, 0.0)]
        reader = _MockReader(metadata=self.metadata, frames=frames)
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("crosshair.crosshair_visible: 0/1 (0%)", warnings_text)
        self.assertIn("kill_feed.kill_feed_active: 0/1 (0%)", warnings_text)
        self.assertIn("money.money_visible: 0/1 (0%)", warnings_text)
        self.assertIn("round_info.round_active: 0/1 (0%)", warnings_text)
        self.assertIn("player_status.hp", warnings_text)
        self.assertIn("player_status.armour", warnings_text)

    def test_crosshair_visible_across_frames(self) -> None:
        frames = [
            _crosshair_frame(0, 0.0),
            _crosshair_frame(1, 0.1),
            _blank_frame(2, 0.2),
        ]
        reader = _MockReader(metadata=self.metadata, frames=frames)
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("crosshair.crosshair_visible: 2/3 (67%)", warnings_text)

    def test_hp_bar_detected_across_frames(self) -> None:
        frames = [_hp_frame(0, 0.0), _blank_frame(1, 0.1)]
        reader = _MockReader(metadata=self.metadata, frames=frames)
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("player_status.armour: 1/2 (50%)", warnings_text)
        self.assertIn("player_status.hp:", warnings_text)

    def test_kill_feed_active_tracking(self) -> None:
        frames = [
            _kill_feed_frame(0, 0.0),
            _kill_feed_frame(1, 0.1),
            _blank_frame(2, 0.2),
            _blank_frame(3, 0.3),
        ]
        reader = _MockReader(metadata=self.metadata, frames=frames)
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("kill_feed.kill_feed_active: 2/4 (50%)", warnings_text)

    def test_full_round_scenario_mixed_frames(self) -> None:
        """Simulate a short round: crosshair + HP + kill feed + round info active."""
        frames = [
            _round_info_frame(0, 0.0),
            _crosshair_frame(1, 0.1),
            _hp_frame(2, 0.2),
            _kill_feed_frame(3, 0.3),
            _money_frame(4, 0.4),
            _blank_frame(5, 0.5),  # round end 鈥?all dark
        ]
        reader = _MockReader(metadata=self.metadata, frames=frames)
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("[pipeline] 6 frames processed", warnings_text)
        self.assertIn("crosshair.crosshair_visible: 1/6 (17%)", warnings_text)
        self.assertIn("kill_feed.kill_feed_active: 1/6 (17%)", warnings_text)
        self.assertIn("money.money_visible: 1/6 (17%)", warnings_text)
        self.assertIn("round_info.round_active: 1/6 (17%)", warnings_text)
        self.assertIn("player_status.armour: 1/6 (17%)", warnings_text)

    def test_metadata_is_preserved_in_result(self) -> None:
        reader = _MockReader(metadata=self.metadata, frames=[_blank_frame(0, 0.0)])
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        self.assertEqual(result.metadata.fps, 60.0)
        self.assertEqual(result.metadata.width, 1920)
        self.assertEqual(result.metadata.height, 1080)
        self.assertEqual(result.metadata.duration_sec, 10.0)
        self.assertEqual(result.metadata.codec, "mock")

    def test_video_identity_preserved(self) -> None:
        reader = _MockReader(metadata=self.metadata, frames=[_blank_frame(0, 0.0)])
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        self.assertEqual(result.video.video_id, "test-001")
        self.assertEqual(result.video.path, Path("test.mp4"))

    def test_reader_error_is_captured_as_warning(self) -> None:
        """When frame iteration raises, the pipeline captures it and continues."""

        class _FailingReader(VideoReader):
            def inspect(self, video: VideoInput) -> VideoMetadata:
                return VideoMetadata(fps=30.0)

            def frames(self, video: VideoInput, sample_fps: float) -> Iterator[VideoFrame]:
                yield _crosshair_frame(0, 0.0)
                raise RuntimeError("simulated read failure after first frame")
                yield  # unreachable

        reader = _FailingReader()
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=10)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("Frame processing error", warnings_text)
        self.assertIn("simulated read failure", warnings_text)

    def test_sample_fps_is_passed_to_reader(self) -> None:
        """Verify the pipeline forwards sample_fps to the reader."""
        captured_fps: list[float] = []

        class _SpyingReader(VideoReader):
            def inspect(self, video: VideoInput) -> VideoMetadata:
                return VideoMetadata(fps=30.0)

            def frames(self, video: VideoInput, sample_fps: float) -> Iterator[VideoFrame]:
                captured_fps.append(sample_fps)
                return iter([])

        reader = _SpyingReader()
        pipeline = VideoAnalysisPipeline(reader, self.parser, sample_fps=5)
        pipeline.run(self.video)

        self.assertEqual(captured_fps, [5.0])


# -- detection integration tests ---------------------------------------------


class _MockDetector(ObjectDetector):
    """Returns fixed detections."""
    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._dets = detections or []
        self.calls: list[tuple] = []

    def detect(self, frame: object, frame_index: int, timestamp_sec: float) -> Sequence[Detection]:
        self.calls.append((frame_index, timestamp_sec))
        return tuple(self._dets)


class _MockClassifier:
    """Returns detections with labels changed."""
    def __init__(self, label_map: dict[str, str] | None = None) -> None:
        self._map = label_map or {}
        self.calls: list[tuple] = []

    def classify(self, frame, detections):
        self.calls.append((len(detections),))
        result = []
        for d in detections:
            new_label = self._map.get(d.label, d.label)
            result.append(Detection(
                label=new_label, confidence=d.confidence,
                bbox_xyxy=d.bbox_xyxy, frame_index=d.frame_index,
                timestamp_sec=d.timestamp_sec,
            ))
        return result


class VideoAnalysisPipelineDetectionTests(TestCase):
    """End-to-end tests with detector and classifier wired in."""

    def setUp(self) -> None:
        self.video = VideoInput(video_id="test-det", path=Path("test.mp4"))
        self.metadata = VideoMetadata(
            fps=60.0, width=1920, height=1080, duration_sec=10.0, codec="mock"
        )
        self.parser = CS2HudParser(CS2_STANDARD_16X9, {})

    def test_no_detector_still_works(self) -> None:
        """Backward compatibility: pipeline without detector runs normally."""
        reader = _MockReader(metadata=self.metadata, frames=[_blank_frame(0, 0.0)])
        pipeline = VideoAnalysisPipeline(reader, self.parser, detector=None)
        result = pipeline.run(self.video)
        self.assertIn("[pipeline] 1 frames processed", result.warnings)

    def test_detector_called_per_frame(self) -> None:
        detector = _MockDetector()
        reader = _MockReader(metadata=self.metadata, frames=[
            _blank_frame(0, 0.0),
            _blank_frame(1, 0.1),
            _blank_frame(2, 0.2),
        ])
        pipeline = VideoAnalysisPipeline(reader, self.parser, detector=detector)
        pipeline.run(self.video)
        self.assertEqual(len(detector.calls), 3)

    def test_detections_reported_in_summary(self) -> None:
        detections = [
            Detection(label="player", confidence=0.9, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0),
            Detection(label="player", confidence=0.8, bbox_xyxy=(300, 200, 400, 400), frame_index=0, timestamp_sec=0.0),
        ]
        detector = _MockDetector(detections=detections)
        reader = _MockReader(metadata=self.metadata, frames=[_blank_frame(0, 0.0)])
        pipeline = VideoAnalysisPipeline(reader, self.parser, detector=detector)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("2 total detections", warnings_text)
        self.assertIn("detections.player: 2", warnings_text)
        self.assertIn("detections.avg_confidence: 0.85", warnings_text)

    def test_classifier_transforms_labels(self) -> None:
        detections = [
            Detection(label="player", confidence=0.9, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0),
            Detection(label="player", confidence=0.8, bbox_xyxy=(300, 200, 400, 400), frame_index=0, timestamp_sec=0.0),
        ]
        detector = _MockDetector(detections=detections)
        classifier = _MockClassifier(label_map={"player": "enemy"})
        reader = _MockReader(metadata=self.metadata, frames=[_blank_frame(0, 0.0)])
        pipeline = VideoAnalysisPipeline(
            reader, self.parser, detector=detector, classifier=classifier,
        )
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("detections.enemy: 2", warnings_text)
        self.assertNotIn("detections.player:", warnings_text)

    def test_empty_detections_no_crash(self) -> None:
        detector = _MockDetector(detections=[])
        reader = _MockReader(metadata=self.metadata, frames=[_blank_frame(0, 0.0)])
        pipeline = VideoAnalysisPipeline(reader, self.parser, detector=detector)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        self.assertIn("0 total detections", warnings_text)

    def test_full_pipeline_hud_and_detection(self) -> None:
        """Both HUD parsing and detection run side by side."""
        detections = [
            Detection(label="player", confidence=0.95, bbox_xyxy=(100, 200, 180, 400), frame_index=0, timestamp_sec=0.0),
        ]
        detector = _MockDetector(detections=detections)
        reader = _MockReader(metadata=self.metadata, frames=[_crosshair_frame(0, 0.0)])
        parser = CS2HudParser(CS2_STANDARD_16X9, {"crosshair": CrosshairExtractor()})
        pipeline = VideoAnalysisPipeline(reader, parser, detector=detector)
        result = pipeline.run(self.video)

        warnings_text = "\n".join(result.warnings)
        # HUD output present
        self.assertIn("crosshair.crosshair_visible", warnings_text)
        # Detection output present
        self.assertIn("1 total detections", warnings_text)
        self.assertIn("detections.player: 1", warnings_text)
