"""Report-generation interfaces and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gamesight.domain.models import AnalysisResult, Track
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.models import MatchReport


class ReportGenerator(ABC):
    @abstractmethod
    def generate(self, analysis: AnalysisResult) -> dict[str, object]:
        """Generate a report using only structured, traceable pipeline evidence."""


class EvidenceReportGenerator(ReportGenerator):
    """Concrete report generator backed by ``EvidenceReportBuilder``.

    Produces a ``MatchReport`` and serialises it to a JSON-safe dict via
    ``model_dump(mode='json')``.  The report is fully evidence-grounded:
    every finding carries explicit ``EvidenceLink`` references back to
    specific frames and pipeline sources.

    Parameters
    ----------
    tracks:
        Optional track list from the detection/tracking pipeline.  When
        provided, per-round track statistics are included in the report.
    """

    def __init__(self, tracks: list[Track] | None = None) -> None:
        self._builder = EvidenceReportBuilder()
        self._tracks = tracks

    def generate(self, analysis: AnalysisResult) -> dict[str, object]:
        report = self._builder.build(analysis, self._tracks)
        return report.model_dump(mode="json")

    def generate_report(self, analysis: AnalysisResult) -> MatchReport:
        """Return the structured ``MatchReport`` object before serialisation."""
        return self._builder.build(analysis, self._tracks)
