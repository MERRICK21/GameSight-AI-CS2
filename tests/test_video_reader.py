"""Unit tests for OpenCV-based video ingestion (metadata + frame sampling)."""

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, call

import numpy as np

from gamesight.domain.models import VideoInput
from gamesight.ingestion.video_reader import (
    FrameSamplingError,
    OpenCVVideoReader,
    VideoReadError,
)


class OpenCVVideoReaderTests(TestCase):
    # -- helpers shared by inspect tests ---------------------------------

    @staticmethod
    def _fourcc(codec: str) -> int:
        return sum(ord(char) << (8 * index) for index, char in enumerate(codec))

    def _backend(self, capture: Mock) -> SimpleNamespace:
        return SimpleNamespace(
            CAP_PROP_FPS=5,
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FOURCC=6,
            CAP_PROP_POS_FRAMES=1,
            VideoCapture=Mock(return_value=capture),
        )

    @staticmethod
    def _capture(
        *,
        opened: bool = True,
        values: dict[int, float] | None = None,
    ) -> Mock:
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.get.side_effect = lambda property_id: (values or {}).get(
            property_id, 0.0
        )
        return capture

    # -- inspect tests (Task 1 regression) -------------------------------

    def test_inspect_reads_video_metadata_and_releases_capture(self) -> None:
        values = {
            5: 60.0, 7: 900.0, 3: 1920.0, 4: 1080.0,
            6: float(self._fourcc("avc1")),
        }
        capture = self._capture(values=values)
        backend = self._backend(capture)
        video = VideoInput(video_id="match-001", path=Path("match.mp4"))

        metadata = OpenCVVideoReader(cv2_module=backend).inspect(video)

        backend.VideoCapture.assert_called_once_with("match.mp4")
        capture.release.assert_called_once()
        self.assertEqual(metadata.fps, 60.0)
        self.assertEqual(metadata.duration_sec, 15.0)
        self.assertEqual(metadata.width, 1920)
        self.assertEqual(metadata.height, 1080)
        self.assertEqual(metadata.codec, "avc1")

    def test_inspect_returns_none_for_missing_fps_and_codec(self) -> None:
        capture = self._capture(
            values={5: 0.0, 7: 900.0, 3: 1280.0, 4: 720.0, 6: 0.0}
        )
        video = VideoInput(video_id="match-002", path=Path("missing-fps.mp4"))

        metadata = OpenCVVideoReader(
            cv2_module=self._backend(capture)
        ).inspect(video)

        self.assertIsNone(metadata.fps)
        self.assertIsNone(metadata.duration_sec)
        self.assertIsNone(metadata.codec)
        capture.release.assert_called_once()

    def test_inspect_raises_for_unopenable_video(self) -> None:
        capture = self._capture(opened=False)
        video = VideoInput(video_id="bad-video", path=Path("bad.mp4"))

        with self.assertRaisesRegex(VideoReadError, "Unable to open video"):
            OpenCVVideoReader(
                cv2_module=self._backend(capture)
            ).inspect(video)

        capture.release.assert_called_once()


class FrameSamplingTests(TestCase):
    """Unit tests for the frames() method -- Sprint 1 Task 2."""

    # -- test helpers ----------------------------------------------------

    @staticmethod
    def _fourcc(codec: str) -> int:
        return sum(ord(char) << (8 * index) for index, char in enumerate(codec))

    @staticmethod
    def _fake_image(width: int = 1920, height: int = 1080) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _backend(self, capture: Mock) -> SimpleNamespace:
        return SimpleNamespace(
            CAP_PROP_FPS=5,
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FOURCC=6,
            CAP_PROP_POS_FRAMES=1,
            VideoCapture=Mock(return_value=capture),
        )

    def _capture_for_sampling(
        self,
        native_fps: float,
        total_frames: int,
    ) -> Mock:
        """Create a mock capture that returns success for every read."""
        capture = Mock()
        capture.isOpened.return_value = True
        values: dict[int, float] = {
            5: native_fps,
            7: float(total_frames),
            3: 1920.0,
            4: 1080.0,
            6: float(self._fourcc("avc1")),
        }
        capture.get.side_effect = lambda prop: values.get(prop, 0.0)
        image = self._fake_image()
        capture.read.return_value = (True, image)
        return capture

    # -- normal sampling -------------------------------------------------

    def test_frames_samples_uniformly_at_lower_rate(self) -> None:
        """60 fps, 300 frames, sample at 10 fps -> 50 frames, step=6."""
        capture = self._capture_for_sampling(
            native_fps=60.0, total_frames=300
        )
        video = VideoInput(video_id="r1", path=Path("r1.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        frames = list(reader.frames(video, sample_fps=10.0))

        self.assertEqual(len(frames), 50)
        self.assertEqual(frames[0].frame_index, 0)
        self.assertEqual(frames[0].timestamp_sec, 0.0)
        self.assertEqual(frames[1].frame_index, 6)
        self.assertEqual(frames[2].frame_index, 12)
        self.assertEqual(frames[-1].frame_index, 294)
        self.assertAlmostEqual(frames[-1].timestamp_sec, 294 / 60.0)

    def test_frames_yields_every_frame_when_sample_exceeds_native(self) -> None:
        """sample_fps > native_fps -> step clamped to 1, every frame yielded."""
        capture = self._capture_for_sampling(
            native_fps=30.0, total_frames=10
        )
        video = VideoInput(video_id="r2", path=Path("r2.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        frames = list(reader.frames(video, sample_fps=120.0))

        self.assertEqual(len(frames), 10)
        self.assertEqual(
            [f.frame_index for f in frames],
            list(range(10)),
        )

    def test_frames_seek_is_called_with_correct_positions(self) -> None:
        """Verify capture.set() is called for each sample position."""
        capture = self._capture_for_sampling(
            native_fps=60.0, total_frames=120
        )
        video = VideoInput(video_id="r3", path=Path("r3.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        list(reader.frames(video, sample_fps=10.0))

        expected_calls = [
            call(1, i * 6) for i in range(20)
        ]
        capture.set.assert_has_calls(expected_calls)

    # -- edge cases ------------------------------------------------------

    def test_frames_raises_when_native_fps_is_unavailable(self) -> None:
        """Missing FPS means we cannot compute a step interval."""
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 0.0
        video = VideoInput(video_id="bad", path=Path("bad.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        with self.assertRaisesRegex(
            FrameSamplingError, "FPS is unavailable"
        ):
            list(reader.frames(video, sample_fps=10.0))

        capture.release.assert_called_once()

    def test_frames_raises_when_sample_fps_is_zero(self) -> None:
        capture = self._capture_for_sampling(
            native_fps=60.0, total_frames=100
        )
        video = VideoInput(video_id="r4", path=Path("r4.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        with self.assertRaisesRegex(
            FrameSamplingError, "sample_fps must be positive"
        ):
            list(reader.frames(video, sample_fps=0.0))

        # Validation raises before VideoCapture is opened.
        capture.release.assert_not_called()

    def test_frames_raises_when_sample_fps_is_negative(self) -> None:
        capture = self._capture_for_sampling(
            native_fps=60.0, total_frames=100
        )
        video = VideoInput(video_id="r5", path=Path("r5.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        with self.assertRaisesRegex(
            FrameSamplingError, "sample_fps must be positive"
        ):
            list(reader.frames(video, sample_fps=-5.0))

        capture.release.assert_not_called()

    def test_frames_yields_nothing_for_zero_frame_video(self) -> None:
        capture = self._capture_for_sampling(
            native_fps=30.0, total_frames=0
        )
        video = VideoInput(video_id="empty", path=Path("empty.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        frames = list(reader.frames(video, sample_fps=10.0))

        self.assertEqual(frames, [])
        capture.release.assert_called_once()

    def test_frames_releases_capture_even_on_error(self) -> None:
        """The finally block must always release the capture."""
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 0.0
        video = VideoInput(video_id="x", path=Path("x.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        try:
            list(reader.frames(video, sample_fps=10.0))
        except FrameSamplingError:
            pass

        capture.release.assert_called_once()

    def test_frames_stops_when_read_fails_mid_stream(self) -> None:
        """If read() returns (False, ...) mid-stream, iteration stops early."""
        capture = Mock()
        capture.isOpened.return_value = True
        values = {5: 60.0, 7: 300.0, 3: 1920.0, 4: 1080.0, 6: 0.0}
        capture.get.side_effect = lambda prop: values.get(prop, 0.0)
        capture.read.side_effect = [
            (True, self._fake_image()),
        ] + [
            (False, None),
        ] * 1000
        video = VideoInput(video_id="broken", path=Path("broken.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        frames = list(reader.frames(video, sample_fps=10.0))

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_index, 0)
        capture.release.assert_called_once()

    def test_frames_image_is_numpy_array(self) -> None:
        """Each yielded frame carries a valid BGR image."""
        capture = self._capture_for_sampling(
            native_fps=30.0, total_frames=3
        )
        video = VideoInput(video_id="r6", path=Path("r6.mp4"))
        reader = OpenCVVideoReader(cv2_module=self._backend(capture))

        frames = list(reader.frames(video, sample_fps=30.0))

        for f in frames:
            self.assertIsInstance(f.image, np.ndarray)
            self.assertEqual(f.image.shape, (1080, 1920, 3))
            self.assertEqual(f.image.dtype, np.uint8)
