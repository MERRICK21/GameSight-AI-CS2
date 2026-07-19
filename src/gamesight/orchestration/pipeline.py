"""Top-level pipeline interface; implementation is intentionally deferred."""

from abc import ABC, abstractmethod

from gamesight.domain.models import AnalysisResult, VideoInput


class AnalysisPipeline(ABC):
    @abstractmethod
    def run(self, video: VideoInput) -> AnalysisResult:
        """Run the configured analysis workflow for one CS2 recording."""
