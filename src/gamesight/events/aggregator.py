"""Event aggregator that groups GameEvents into RoundAnalysis objects."""

from __future__ import annotations

from collections.abc import Iterable

from gamesight.domain.models import EventType, GameEvent, RoundAnalysis


def aggregate_events(
    events: Iterable[GameEvent],
) -> list[RoundAnalysis]:
    """Sort events by time and group them into per-round analysis objects.

    Rounds are bounded by paired ``ROUND_START`` / ``ROUND_END`` events.
    Events that fall between a start and end are assigned to that round.
    Orphan events (outside any round boundary) are silently discarded.

    Returns rounds in chronological order.
    """
    sorted_events = sorted(events, key=lambda e: e.start_sec)
    rounds: list[RoundAnalysis] = []
    current_round: RoundAnalysis | None = None

    for event in sorted_events:
        if event.event_type == EventType.ROUND_START:
            # Close any dangling round before opening a new one.
            if current_round is not None:
                rounds.append(current_round)
            rid = event.attributes.get("round_id", "unknown") if event.attributes else "unknown"
            current_round = RoundAnalysis(
                round_id=str(rid),
                start_sec=event.start_sec,
            )
            current_round.events.append(event)

        elif event.event_type == EventType.ROUND_END:
            if current_round is not None:
                current_round.end_sec = event.start_sec
                current_round.events.append(event)
                rounds.append(current_round)
                current_round = None
            # ROUND_END without a preceding ROUND_START → discard.

        else:
            # Non-boundary event (kill, death, etc.)
            if current_round is not None:
                current_round.events.append(event)
            # Orphan non-boundary events → discard.

    # Close any round that never received a ROUND_END (truncated video).
    if current_round is not None:
        rounds.append(current_round)

    return rounds
