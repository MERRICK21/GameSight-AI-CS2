"""Tracker contract for future ByteTrack or BoT-SORT adapters."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from gamesight.domain.models import Detection, Track


class MultiObjectTracker(ABC):
    @abstractmethod
    def update(self, detections: Sequence[Detection]) -> Sequence[Track]:
        """Associate current detections with persistent object trajectories."""

    @abstractmethod
    def reset(self) -> None:
        """Clear tracker state at a video or round boundary."""
