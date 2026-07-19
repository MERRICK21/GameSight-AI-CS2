"""Video ingestion contract; decoding is implemented in a later sprint."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from gamesight.domain.models import VideoInput, VideoMetadata


class VideoReader(ABC):
    @abstractmethod
    def inspect(self, video: VideoInput) -> VideoMetadata:
        """Read metadata and validate a video without changing it."""

    @abstractmethod
    def frames(self, video: VideoInput, sample_fps: float) -> Iterator[object]:
        """Yield decoded frame records at the requested analysis rate."""
