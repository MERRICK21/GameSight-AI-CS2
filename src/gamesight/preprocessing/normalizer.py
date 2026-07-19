"""Interfaces for resolution, aspect-ratio, and quality normalization."""

from abc import ABC, abstractmethod

from gamesight.domain.models import VideoInput, VideoMetadata


class Preprocessor(ABC):
    @abstractmethod
    def assess_quality(self, video: VideoInput, metadata: VideoMetadata) -> dict[str, object]:
        """Return quality and domain-shift diagnostics for the source video."""

    @abstractmethod
    def normalize(self, video: VideoInput, metadata: VideoMetadata) -> VideoInput:
        """Return a normalized analysis copy or a reference to the original input."""
