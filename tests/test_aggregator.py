"""Unit tests for event aggregator."""

from unittest import TestCase

from gamesight.domain.models import EventType, Evidence, GameEvent, RoundAnalysis
from gamesight.events.aggregator import aggregate_events


def _event(
    event_type: EventType,
    ts: float,
    round_id: str | None = None,
    event_id: str | None = None,
) -> GameEvent:
    attrs = {}
    if round_id is not None:
        attrs["round_id"] = round_id
    return GameEvent(
        event_id=event_id or f"{event_type.value}_test",
        event_type=event_type,
        start_sec=ts,
        confidence=0.9,
        evidence=[Evidence(timestamp_sec=ts, source="test")],
        attributes=attrs,
    )


class AggregateEventsBasicTests(TestCase):
    """Happy-path and empty-input tests."""

    def test_empty_returns_empty(self) -> None:
        result = aggregate_events([])
        self.assertEqual(result, [])

    def test_single_complete_round(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.PLAYER_KILL, 5.0),
            _event(EventType.PLAYER_DEATH, 8.0),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)

        r = result[0]
        self.assertEqual(r.round_id, "round_001")
        self.assertEqual(r.start_sec, 0.0)
        self.assertEqual(r.end_sec, 10.0)
        self.assertEqual(len(r.events), 4)

        event_types = [e.event_type for e in r.events]
        self.assertEqual(
            event_types,
            [EventType.ROUND_START, EventType.PLAYER_KILL, EventType.PLAYER_DEATH, EventType.ROUND_END],
        )

    def test_multiple_rounds(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.PLAYER_KILL, 3.0),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
            _event(EventType.ROUND_START, 20.0, round_id="round_002"),
            _event(EventType.PLAYER_DEATH, 25.0),
            _event(EventType.PLAYER_KILL, 28.0),
            _event(EventType.ROUND_END, 35.0, round_id="round_002"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 2)

        self.assertEqual(result[0].round_id, "round_001")
        self.assertEqual(result[0].start_sec, 0.0)
        self.assertEqual(result[0].end_sec, 10.0)
        self.assertEqual(len(result[0].events), 3)

        self.assertEqual(result[1].round_id, "round_002")
        self.assertEqual(result[1].start_sec, 20.0)
        self.assertEqual(result[1].end_sec, 35.0)
        self.assertEqual(len(result[1].events), 4)

    def test_events_sorted_by_time(self) -> None:
        """Events arriving out of order should be sorted."""
        events = [
            _event(EventType.PLAYER_KILL, 5.0),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].round_id, "round_001")
        # Events within the round should be in chronological order
        self.assertEqual(result[0].events[0].event_type, EventType.ROUND_START)
        self.assertEqual(result[0].events[1].event_type, EventType.PLAYER_KILL)
        self.assertEqual(result[0].events[2].event_type, EventType.ROUND_END)


class AggregateEventsEdgeCaseTests(TestCase):
    """Boundary and orphan event tests."""

    def test_orphan_events_before_first_round_discarded(self) -> None:
        events = [
            _event(EventType.PLAYER_KILL, 1.0),  # orphan
            _event(EventType.ROUND_START, 5.0, round_id="round_001"),
            _event(EventType.ROUND_END, 15.0, round_id="round_001"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        # The round should only contain start + end, not the orphan kill
        self.assertEqual(len(result[0].events), 2)

    def test_orphan_events_between_rounds_discarded(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
            _event(EventType.PLAYER_KILL, 15.0),  # between rounds
            _event(EventType.ROUND_START, 20.0, round_id="round_002"),
            _event(EventType.ROUND_END, 30.0, round_id="round_002"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].events), 2)
        self.assertEqual(len(result[1].events), 2)

    def test_orphan_events_after_last_round_discarded(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
            _event(EventType.PLAYER_DEATH, 12.0),  # after last round
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].events), 2)

    def test_round_end_without_start_discarded(self) -> None:
        events = [
            _event(EventType.ROUND_END, 5.0, round_id="round_001"),
            _event(EventType.ROUND_START, 10.0, round_id="round_001"),
            _event(EventType.ROUND_END, 20.0, round_id="round_001"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].round_id, "round_001")
        self.assertEqual(result[0].start_sec, 10.0)

    def test_truncated_round_no_end(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.PLAYER_KILL, 3.0),
            _event(EventType.PLAYER_DEATH, 7.0),
            # No ROUND_END 閳?video truncated mid-round
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].round_id, "round_001")
        self.assertIsNone(result[0].end_sec)
        self.assertEqual(len(result[0].events), 3)

    def test_consecutive_round_starts_use_latest(self) -> None:
        """A second ROUND_START without an intervening ROUND_END starts a new round."""
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.PLAYER_KILL, 3.0),
            _event(EventType.ROUND_START, 5.0, round_id="round_002"),  # new round
            _event(EventType.PLAYER_DEATH, 8.0),
            _event(EventType.ROUND_END, 15.0, round_id="round_002"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 2)

        # Round 1: truncated, has no end
        self.assertEqual(result[0].round_id, "round_001")
        self.assertIsNone(result[0].end_sec)
        self.assertEqual(len(result[0].events), 2)  # start + kill

        # Round 2: complete
        self.assertEqual(result[1].round_id, "round_002")
        self.assertEqual(result[1].end_sec, 15.0)
        self.assertEqual(len(result[1].events), 3)  # start + death + end

    def test_round_with_no_non_boundary_events(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].events), 2)
        self.assertIsNone(result[0].events[0].end_sec)  # ROUND_START has no end_sec

    def test_mixed_event_types_preserved(self) -> None:
        events = [
            _event(EventType.ROUND_START, 0.0, round_id="round_001"),
            _event(EventType.PLAYER_KILL, 2.0, event_id="kill_001"),
            _event(EventType.PLAYER_DEATH, 4.0, event_id="death_001"),
            _event(EventType.PLAYER_KILL, 6.0, event_id="kill_002"),
            _event(EventType.ENEMY_FIRST_VISIBLE, 7.0, event_id="efv_001"),
            _event(EventType.ROUND_END, 10.0, round_id="round_001"),
        ]
        result = aggregate_events(events)
        self.assertEqual(len(result), 1)
        types_in_round = [e.event_type for e in result[0].events]
        self.assertIn(EventType.PLAYER_KILL, types_in_round)
        self.assertIn(EventType.PLAYER_DEATH, types_in_round)
        self.assertIn(EventType.ENEMY_FIRST_VISIBLE, types_in_round)
        self.assertEqual(len(result[0].events), 6)
