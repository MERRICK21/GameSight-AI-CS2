"""Evidence-grounded report data models.

Every finding in the report carries explicit evidence links that trace
back to specific frames, timestamps, and pipeline sources.  This makes
the report auditable — whether the narrative text comes from an LLM or
from a deterministic template, every claim has a paper trail.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceLink(BaseModel):
    """A single piece of evidence backing one report finding.

    Mirrors the domain ``Evidence`` model but uses only lightweight
    scalar fields so that the report dict is fully serialisable.
    """

    frame_index: int | None = None
    timestamp_sec: float
    source: str
    asset_path: str | None = None


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FindingCategory(StrEnum):
    COMBAT = "combat"
    MOVEMENT = "movement"
    UTILITY = "utility"
    TEAMPLAY = "teamplay"
    ROUND_FLOW = "round_flow"


class ReportFinding(BaseModel):
    """One evidence-grounded finding within a round or match-level section.

    ``text`` carries the human-readable explanation.  ``evidence`` links
    back to specific pipeline data points (frame, timestamp, source) so
    the reader can verify every claim.
    """

    finding_id: str
    category: FindingCategory
    severity: FindingSeverity = FindingSeverity.INFO
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoundStats(BaseModel):
    """Quantitative summary for one round, computed from pipeline events."""

    round_id: str
    duration_sec: float | None = None
    kills_detected: int = 0
    player_died: bool = False
    deaths_detected: int = 0
    killfeed_events: int = 0
    enemy_tracks: int = 0
    teammate_tracks: int = 0
    enemy_first_visible_sec: float | None = None
    combat_segments: int = 0


class RoundReport(BaseModel):
    """Per-round section of the evidence report."""

    round_id: str
    start_sec: float
    end_sec: float | None = None
    duration_sec: float | None = None
    stats: RoundStats
    findings: list[ReportFinding] = Field(default_factory=list)


class MatchOverview(BaseModel):
    """Top-level match summary included in every report."""

    video_id: str
    source_name: str | None = None
    duration_sec: float | None = None
    fps: float | None = None
    resolution: dict[str, int] = Field(default_factory=dict)
    total_rounds: int = 0
    total_kills_detected: int = 0
    total_deaths_detected: int = 0
    total_enemy_tracks: int = 0
    warnings: list[str] = Field(default_factory=list)


class MatchReport(BaseModel):
    """Complete evidence-grounded match report.

    ``generated_at`` is set to the current UTC time on construction so that
    every report carries a creation timestamp without caller effort.
    """

    report_version: str = "1.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overview: MatchOverview
    rounds: list[RoundReport] = Field(default_factory=list)
    match_findings: list[ReportFinding] = Field(default_factory=list)

    def model_dump_for_json(self) -> dict[str, Any]:
        """Return a JSON-safe dict, with datetime serialised as ISO-8601."""
        return self.model_dump(mode="json")
