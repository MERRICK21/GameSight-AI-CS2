"""LLM-independent report-generation interface."""

from abc import ABC, abstractmethod

from gamesight.domain.models import AnalysisResult


class ReportGenerator(ABC):
    @abstractmethod
    def generate(self, analysis: AnalysisResult) -> dict[str, object]:
        """Generate a report using only structured, traceable pipeline evidence."""
