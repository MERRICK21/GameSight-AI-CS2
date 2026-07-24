"""Timeline data models — flattened, JSON-serializable representations.

These models decouple the serialization layer from the internal domain
models (``GameEvent``, ``RoundAnalysis``, ``AnalysisResult``) so that
the export format can evolve independently of the analysis pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from gamesight.domain.models import AnalysisResult, RoundAnalysis, Track


class EvidenceRef(BaseModel):
    """Lightweight evidence pointer for JSON export.

    Carries only enough information to trace a timeline entry back to the
    source frame, without duplicating large asset payloads.
    """

    frame_index: int | None = None
    timestamp_sec: float
    source: str


class TimelineEvent(BaseModel):
    """A single event flattened for timeline JSON export.

    Mirrors ``GameEvent`` but drops the nested ``Evidence`` objects in
    favour of ``EvidenceRef`` and flattens ``EventType`` to a plain string
    so consumers do not need the enum definition.
    """

    event_id: str
    event_type: str
    start_sec: float
    end_sec: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    round_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class TrackSummary(BaseModel):
    """Summarised track for timeline export.

    Full per-frame ``Detection`` lists are omitted to keep the JSON
    compact; consumers that need the raw data can cross-reference
    with the detection artifact sidecar.
    """

    track_id: str
    label: str
    first_seen_sec: float
    last_seen_sec: float
    detection_count: int
    avg_confidence: float = Field(ge=0.0, le=1.0)


class RoundTimeline(BaseModel):
    """One round's worth of timeline data — events + track summaries."""

    round_id: str
    start_sec: float
    end_sec: float | None = None
    duration_sec: float | None = None
    events: list[TimelineEvent] = Field(default_factory=list)
    tracks: list[TrackSummary] = Field(default_factory=list)


class MatchTimeline(BaseModel):
    """Complete match timeline, ready for JSON serialisation."""

    schema_version: str = "2.0"
    video_id: str
    source_name: str | None = None
    duration_sec: float | None = None
    fps: float | None = None
    resolution: dict[str, int] = Field(default_factory=dict)
    total_rounds: int = 0
    rounds: list[RoundTimeline] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TimelineBuilder:
    """Convert internal analysis data into a serialisable MatchTimeline.

    Consumes an ``AnalysisResult`` (which carries rounds + metadata) and an
    optional track list from the pipeline, then produces a flat,
    JSON-ready ``MatchTimeline`` with per-round event and track summaries.
    """

    def build(
        self,
        analysis: AnalysisResult,
        tracks: list[Track] | None = None,
    ) -> MatchTimeline:
        """Build a MatchTimeline from pipeline outputs."""
        # Deduplicate tracks (tracker returns same tracks each frame).
        unique_tracks: dict[str, Track] = {}
        for t in (tracks or []):
            existing = unique_tracks.get(t.track_id)
            if existing is None or len(t.detections) > len(existing.detections):
                unique_tracks[t.track_id] = t

        track_summaries = [self._summarise_track(t) for t in unique_tracks.values()]

        round_timelines: list[RoundTimeline] = []
        for ra in analysis.rounds:
            rt = self._build_round(ra, track_summaries)
            round_timelines.append(rt)

        return MatchTimeline(
            schema_version="2.0",
            video_id=analysis.video.video_id,
            source_name=analysis.video.source_name,
            duration_sec=analysis.metadata.duration_sec,
            fps=analysis.metadata.fps,
            resolution={
                "width": analysis.metadata.width or 0,
                "height": analysis.metadata.height or 0,
            },
            total_rounds=len(round_timelines),
            rounds=round_timelines,
            warnings=list(analysis.warnings),
        )

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _build_round(
        ra: RoundAnalysis,
        all_tracks: list[TrackSummary],
    ) -> RoundTimeline:
        """Convert one RoundAnalysis into a RoundTimeline."""
        events = [TimelineBuilder._convert_event(e, ra.round_id) for e in ra.events]

        # Assign tracks that overlap this round's time window.
        round_tracks = [
            t for t in all_tracks
            if TimelineBuilder._track_overlaps_round(t, ra.start_sec, ra.end_sec)
        ]

        duration = None
        if ra.end_sec is not None:
            duration = round(ra.end_sec - ra.start_sec, 2)

        return RoundTimeline(
            round_id=ra.round_id,
            start_sec=ra.start_sec,
            end_sec=ra.end_sec,
            duration_sec=duration,
            events=events,
            tracks=round_tracks,
        )

    @staticmethod
    def _convert_event(e, round_id: str) -> TimelineEvent:
        """Flatten a GameEvent into a TimelineEvent."""
        evidence_refs = [
            EvidenceRef(
                frame_index=ev.frame_index,
                timestamp_sec=ev.timestamp_sec,
                source=ev.source,
            )
            for ev in e.evidence
        ]
        return TimelineEvent(
            event_id=e.event_id,
            event_type=e.event_type.value,
            start_sec=e.start_sec,
            end_sec=e.end_sec,
            confidence=e.confidence,
            round_id=round_id,
            attributes=dict(e.attributes),
            evidence=evidence_refs,
        )

    @staticmethod
    def _summarise_track(t: Track) -> TrackSummary:
        """Build a TrackSummary from a Track."""
        dets = t.detections
        if not dets:
            return TrackSummary(
                track_id=t.track_id,
                label=t.label,
                first_seen_sec=0.0,
                last_seen_sec=0.0,
                detection_count=0,
                avg_confidence=0.0,
            )
        sorted_dets = sorted(dets, key=lambda d: d.timestamp_sec)
        avg_conf = round(sum(d.confidence for d in dets) / len(dets), 4)
        return TrackSummary(
            track_id=t.track_id,
            label=t.label,
            first_seen_sec=sorted_dets[0].timestamp_sec,
            last_seen_sec=sorted_dets[-1].timestamp_sec,
            detection_count=len(dets),
            avg_confidence=avg_conf,
        )

    @staticmethod
    def _track_overlaps_round(
        t: TrackSummary,
        round_start: float,
        round_end: float | None,
    ) -> bool:
        """True when a track's lifetime overlaps the round time window."""
        if round_end is None:
            return t.first_seen_sec >= round_start
        # Overlap: track starts before round end AND track ends after round start.
        return t.first_seen_sec < round_end and t.last_seen_sec >= round_start
