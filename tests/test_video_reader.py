"""Unit tests for metadata-only OpenCV video ingestion."""

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from gamesight.domain.models import VideoInput
from gamesight.ingestion.video_reader import OpenCVVideoReader, VideoReadError


class OpenCVVideoReaderTests(TestCase):
    def _fourcc(self, codec: str) -> int:
        return sum(ord(char) << (8 * index) for index, char in enumerate(codec))

    def _backend(self, capture: Mock) -> SimpleNamespace:
        return SimpleNamespace(
            CAP_PROP_FPS=5,
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FOURCC=6,
            VideoCapture=Mock(return_value=capture),
        )

    def _capture(self, *, opened: bool = True, values: dict[int, float] | None = None) -> Mock:
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.get.side_effect = lambda property_id: (values or {}).get(property_id, 0.0)
        return capture

    def test_inspect_reads_video_metadata_and_releases_capture(self) -> None:
        values = {5: 60.0, 7: 900.0, 3: 1920.0, 4: 1080.0, 6: float(self._fourcc("avc1"))}
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

    def test_inspect_returns_unknown_duration_when_fps_is_unavailable(self) -> None:
        capture = self._capture(values={5: 0.0, 7: 900.0, 3: 1280.0, 4: 720.0, 6: 0.0})
        video = VideoInput(video_id="match-002", path=Path("missing-fps.mp4"))

        metadata = OpenCVVideoReader(cv2_module=self._backend(capture)).inspect(video)

        self.assertIsNone(metadata.fps)
        self.assertIsNone(metadata.duration_sec)
        self.assertIsNone(metadata.codec)
        capture.release.assert_called_once()

    def test_inspect_raises_a_clear_error_for_an_unopenable_video(self) -> None:
        capture = self._capture(opened=False)
        video = VideoInput(video_id="bad-video", path=Path("bad.mp4"))

        with self.assertRaisesRegex(VideoReadError, "Unable to open video: bad.mp4"):
            OpenCVVideoReader(cv2_module=self._backend(capture)).inspect(video)

        capture.release.assert_called_once()
