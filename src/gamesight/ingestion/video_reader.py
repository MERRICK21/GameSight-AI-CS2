"""Video metadata reading and frame sampling backed by OpenCV."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module

from numpy.typing import NDArray

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


class FrameSamplingError(RuntimeError):
    """Raised when frame sampling cannot proceed (e.g. missing FPS)."""


@dataclass(frozen=True)
class VideoFrame:
    """A single decoded frame yielded by the ingestion layer.

    The image is stored as a BGR numpy array (OpenCV native format).
    Downstream modules that need RGB should convert explicitly.
    """

    frame_index: int
    timestamp_sec: float
    image: NDArray


class OpenCVVideoReader(VideoReader):
    """Read video metadata and sample frames via OpenCV.

    Dependency injection: pass ``cv2_module`` to inject a mock backend for
    testing. In production leave it ``None`` to import the real ``cv2``.
    """

    def __init__(self, cv2_module: object | None = None) -> None:
        self._cv2 = cv2_module if cv2_module is not None else import_module("cv2")

    # -- metadata inspection --------------------------------------------------------

    def inspect(self, video: VideoInput) -> VideoMetadata:
        """Return metadata reported by OpenCV for ``video``.

        Duration is derived from frame count divided by FPS. Containers that
        do not expose a valid FPS report ``None`` for duration rather than
        raising or returning an invalid value.
        """
        capture = self._cv2.VideoCapture(str(video.path))
        try:
            if not capture.isOpened():
                raise VideoReadError(f"Unable to open video: {video.path}")

            fps = self._positive_float(capture.get(self._cv2.CAP_PROP_FPS))
            frame_count = self._positive_float(capture.get(self._cv2.CAP_PROP_FRAME_COUNT))

            return VideoMetadata(
                fps=fps,
                duration_sec=(
                    (frame_count / fps)
                    if fps is not None and frame_count is not None
                    else None
                ),
                width=self._positive_int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
                height=self._positive_int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=self._decode_fourcc(capture.get(self._cv2.CAP_PROP_FOURCC)),
            )
        finally:
            capture.release()

    # -- frame sampling ------------------------------------------------------------

    def frames(self, video: VideoInput, sample_fps: float) -> Iterator[VideoFrame]:
        """Yield decoded frames at a uniform rate close to ``sample_fps``.

        Uses sequential reading with ``grab()`` for skipping to avoid the
        slow ``CAP_PROP_POS_FRAMES`` seek, which is unreliable with
        inter-frame compressed codecs (h264/h265).

        Raises ``FrameSamplingError`` when the video does not report a valid
        native FPS, making uniform sampling impossible.
        """
        if sample_fps <= 0:
            raise FrameSamplingError(
                f"sample_fps must be positive, got {sample_fps}"
            )

        capture = self._cv2.VideoCapture(str(video.path))
        try:
            if not capture.isOpened():
                raise VideoReadError(f"Unable to open video: {video.path}")

            native_fps = self._positive_float(
                capture.get(self._cv2.CAP_PROP_FPS)
            )
            if native_fps is None:
                raise FrameSamplingError(
                    f"Video FPS is unavailable; cannot compute sample interval"
                )

            total_frames = self._positive_int(
                capture.get(self._cv2.CAP_PROP_FRAME_COUNT)
            )
            if total_frames is None or total_frames == 0:
                return  # empty video -- yield nothing

            # Step in native frame units.  Clamp to at least 1 so we never
            # get stuck re-reading the same frame when sample_fps > native_fps.
            step = max(1, round(native_fps / sample_fps))

            # Sequential read with frame skipping: grab() only parses frame
            # headers without full decode, much faster than seek+read for
            # compressed codecs.
            target_frame = 0
            current_frame = 0
            while current_frame < total_frames:
                if current_frame == target_frame:
                    success, image = capture.read()
                    if not success:
                        break
                    yield VideoFrame(
                        frame_index=current_frame,
                        timestamp_sec=current_frame / native_fps,
                        image=image,
                    )
                    target_frame += step
                    current_frame += 1
                else:
                    # Skip frame without full decode
                    capture.grab()
                    current_frame += 1
        finally:
            capture.release()

    # -- helpers -------------------------------------------------------------------

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

        codec = "".join(
            chr((code >> (8 * offset)) & 0xFF) for offset in range(4)
        ).rstrip("\x00")
        return codec or None
