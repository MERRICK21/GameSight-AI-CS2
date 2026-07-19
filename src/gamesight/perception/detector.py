"""Object-detector contract for future YOLO-backed implementations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from gamesight.domain.models import Detection


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: object, frame_index: int, timestamp_sec: float) -> Sequence[Detection]:
        """Return enemy/teammate detections for one decoded frame."""
