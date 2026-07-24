"""Evidence-grounded report-generation contracts and implementations.

Public API
----------
ReportGenerator (ABC)
    Abstract interface for all report generators.
EvidenceReportGenerator
    Concrete implementation backed by ``EvidenceReportBuilder``.
MatchReport / RoundReport / ReportFinding / EvidenceLink
    Pydantic models for the structured evidence report.
EvidenceReportBuilder
    Builds a ``MatchReport`` from ``AnalysisResult`` + tracks.
"""

from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.generator import EvidenceReportGenerator, ReportGenerator
from gamesight.reporting.models import (
    EvidenceLink,
    FindingCategory,
    FindingSeverity,
    MatchOverview,
    MatchReport,
    ReportFinding,
    RoundReport,
    RoundStats,
)

__all__ = [
    "EvidenceLink",
    "EvidenceReportBuilder",
    "EvidenceReportGenerator",
    "FindingCategory",
    "FindingSeverity",
    "MatchOverview",
    "MatchReport",
    "ReportFinding",
    "ReportGenerator",
    "RoundReport",
    "RoundStats",
]
