"""User-auditable corrections for personal kill/death detections."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from gamesight.domain.models import AnalysisResult, EventType, Evidence, GameEvent


CORRECTABLE_TYPES = {EventType.PLAYER_KILL, EventType.PLAYER_DEATH}


def apply_event_corrections(
    analysis: AnalysisResult, corrections: Mapping[str, bool]
) -> AnalysisResult:
    """Return a deep copy with explicitly rejected combat events removed."""
    corrected = analysis.model_copy(deep=True)
    for round_analysis in corrected.rounds:
        round_analysis.events = [
            event for event in round_analysis.events
            if event.event_type not in CORRECTABLE_TYPES
            or corrections.get(event.event_id, True)
        ]
    corrected.analysis_metadata["manual_corrections"] = sum(
        not accepted for accepted in corrections.values()
    )
    return corrected


def add_manual_events(
    analysis: AnalysisResult, labels: list[dict[str, object]]
) -> AnalysisResult:
    """Add user-labelled missed events while keeping detector output immutable."""
    augmented = analysis.model_copy(deep=True)
    by_round = {item.round_id: item for item in augmented.rounds}
    for label in labels:
        round_id = str(label["round_id"])
        round_analysis = by_round.get(round_id)
        if round_analysis is None:
            continue
        event_type = EventType(str(label["event_type"]))
        if event_type not in CORRECTABLE_TYPES:
            continue
        timestamp = float(label["timestamp_sec"])
        event_id = str(label["event_id"])
        if any(event.event_id == event_id for event in round_analysis.events):
            continue
        round_analysis.events.append(GameEvent(
            event_id=event_id,
            event_type=event_type,
            start_sec=timestamp,
            confidence=1.0,
            evidence=[Evidence(
                timestamp_sec=timestamp,
                source="UserCorrection.manual_label",
            )],
            attributes={
                "round_id": round_id,
                "method": "manual_user_label",
                "classification": "user_confirmed",
            },
        ))
        round_analysis.events.sort(key=lambda event: event.start_sec)
    augmented.analysis_metadata["manual_additions"] = len(labels)
    return augmented


def diagnostic_rows(
    analysis: AnalysisResult, corrections: Mapping[str, bool]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_analysis in analysis.rounds:
        for event in round_analysis.events:
            if event.event_type not in CORRECTABLE_TYPES:
                continue
            rows.append({
                "event_id": event.event_id,
                "round_id": round_analysis.round_id,
                "event_type": event.event_type.value,
                "timestamp_sec": event.start_sec,
                "confidence": event.confidence,
                "source": event.evidence[0].source if event.evidence else "",
                "method": event.attributes.get("method", ""),
                "accepted": corrections.get(event.event_id, True),
            })
    return rows


def build_correction_export(
    analysis: AnalysisResult, corrections: Mapping[str, bool]
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "video_id": analysis.video.video_id,
        "source_name": analysis.video.source_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "labels": diagnostic_rows(analysis, corrections),
    }
