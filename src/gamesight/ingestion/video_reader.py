"""Video metadata reading backed by OpenCV.

Frame sampling intentionally remains outside Sprint 1 Task 1. This module only
opens a video long enough to read its container-exposed metadata.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from importlib import import_module

from gamesight.domain.models import VideoInput, VideoMetadata


class VideoReader(ABC):
    @abstractmethod
    def inspect(self, video: VideoInput) -> VideoMetadata:
        """Read metadata and validate a video without changing it."""

    @abstractmethod
    def frames(self, video: VideoInput, sample_fps: float) -> Iterator[object]:
        """Yield decoded frame records at the requested analysis rate."""


class VideoReadError(RuntimeError):
    """Raised when OpenCV cannot open a video for metadata inspection."""


class OpenCVVideoReader(VideoReader):
    """Read FPS, duration, dimensions, and codec without decoding all frames."""

    def __init__(self, cv2_module: object | None = None) -> None:
        self._cv2 = cv2_module if cv2_module is not None else import_module("cv2")

    def inspect(self, video: VideoInput) -> VideoMetadata:
        """Return metadata reported by OpenCV for ``video``.

        Duration is derived from frame count divided by FPS. Containers that do
        not expose a valid FPS report ``None`` for duration rather than raising
        or returning an invalid value.
        """
        capture = self._cv2.VideoCapture(str(video.path))
        try:
            if not capture.isOpened():
                raise VideoReadError(f"Unable to open video: {video.path}")

            fps = self._positive_float(capture.get(self._cv2.CAP_PROP_FPS))
            frame_count = self._positive_float(capture.get(self._cv2.CAP_PROP_FRAME_COUNT))

            return VideoMetadata(
                fps=fps,
                duration_sec=(frame_count / fps) if fps is not None and frame_count is not None else None,
                width=self._positive_int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
                height=self._positive_int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=self._decode_fourcc(capture.get(self._cv2.CAP_PROP_FOURCC)),
            )
        finally:
            capture.release()

    def frames(self, video: VideoInput, sample_fps: float) -> Iterator[object]:
        """Reserve frame sampling for a later Sprint 1 task."""
        raise NotImplementedError("Frame sampling is not part of Sprint 1 Task 1.")

    @staticmethod
    def _positive_float(value: float) -> float | None:
        return float(value) if value > 0 else None

    @staticmethod
    def _positive_int(value: float) -> int | None:
        return int(value) if value > 0 else None

    @staticmethod
    def _decode_fourcc(value: float) -> str | None:
        """Decode OpenCV's little-endian FourCC integer into a codec label."""
        code = int(value)
        if code <= 0:
            return None

        codec = "".join(chr((code >> (8 * offset)) & 0xFF) for offset in range(4)).rstrip("\x00")
        return codec or None
