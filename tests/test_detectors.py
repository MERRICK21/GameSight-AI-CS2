"""Unit tests for event detectors."""

from unittest import TestCase

from gamesight.domain.models import EventType, HudState
from gamesight.events.detectors import KillEventDetector, RoundBoundaryDetector
from gamesight.events.engine import EventEngine


def _state(
    fi: int,
    ts: float,
    active: bool,
    key: str = "round_info.round_active",
) -> HudState:
    """Factory for a minimal HudState with a round_active flag."""
    return HudState(
        frame_index=fi,
        timestamp_sec=ts,
        profile="cs2_standard_16x9",
        values={key: active},
        confidence=0.8,
    )


def _assert_event(
    test: TestCase,
    event,
    expected_type: EventType,
    expected_round_id: str,
) -> None:
    """Assert that *event* matches expected type and round id."""
    test.assertEqual(event.event_type, expected_type)
    attrs = event.attributes
    test.assertIsNotNone(attrs)
    test.assertEqual(attrs.get("round_id"), expected_round_id)


# -- RoundBoundaryDetector ---------------------------------------------------

class RoundBoundaryDetectorInterfaceTests(TestCase):
    """Contract and constructor tests."""

    def test_implements_event_engine(self) -> None:
        detector = RoundBoundaryDetector()
        self.assertIsInstance(detector, EventEngine)

    def test_default_constructor_values(self) -> None:
        detector = RoundBoundaryDetector()
        self.assertEqual(detector._state_key, "round_info.round_active")
        self.assertEqual(detector._debounce, 8)
        self.assertEqual(detector._min_duration, 15.0)

    def test_custom_state_key(self) -> None:
        detector = RoundBoundaryDetector(state_key="custom.round_flag")
        self.assertEqual(detector._state_key, "custom.round_flag")

    def test_custom_debounce_frames(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=5)
        self.assertEqual(detector._debounce, 5)

    def test_raises_on_invalid_debounce(self) -> None:
        with self.assertRaises(ValueError):
            RoundBoundaryDetector(debounce_frames=0)

    def test_raises_on_negative_min_duration(self) -> None:
        with self.assertRaises(ValueError):
            RoundBoundaryDetector(min_round_duration_sec=-1.0)

    def test_update_returns_sequence(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2)
        result = detector.update(_state(0, 0.0, False))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 0)


class RoundBoundaryDetectorBasicTests(TestCase):
    """Single-round life-cycle tests."""

    def test_complete_round_lifecycle(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2, min_round_duration_sec=0.0)
        events: list = []

        # Freeze time --- no events
        for i in range(3):
            events += detector.update(_state(i, i * 0.5, False))
        self.assertEqual(len(events), 0)

        # Round starts --- 2 debounce frames -> ROUND_START
        events += detector.update(_state(3, 1.5, True))
        self.assertEqual(len(events), 0, "First active frame should not trigger yet")
        events += detector.update(_state(4, 2.0, True))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")

        # Round active --- no events
        for i in range(5, 20):
            events += detector.update(_state(i, i * 0.5, True))
        start_count = sum(1 for e in events if e.event_type == EventType.ROUND_START)
        self.assertEqual(start_count, 1, "No extra starts during active round")

        # Round ends --- 2 debounce frames -> ROUND_END
        events += detector.update(_state(20, 10.0, False))
        end_before = sum(1 for e in events if e.event_type == EventType.ROUND_END)
        self.assertEqual(end_before, 0)
        events += detector.update(_state(21, 10.5, False))
        self.assertEqual(len(events), 2)
        _assert_event(self, events[1], EventType.ROUND_END, "round_001")

        # Back to freeze --- no events
        events += detector.update(_state(22, 11.0, False))
        self.assertEqual(len(events), 2)

    def test_round_start_timestamp_is_first_active_frame(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=3)
        events: list = []

        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.1, True))
        events += detector.update(_state(2, 0.2, True))

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start_sec, 0.0,
                               msg="Start time should be first active frame")


class RoundBoundaryDetectorDebounceTests(TestCase):
    """Tests that the debounce window suppresses noise."""

    def setUp(self) -> None:
        self.detector = RoundBoundaryDetector(debounce_frames=3)

    def test_flicker_within_debounce_window_suppressed(self) -> None:
        events: list = []
        events += self.detector.update(_state(0, 0.0, False))
        events += self.detector.update(_state(1, 0.5, True))
        events += self.detector.update(_state(2, 1.0, False))
        events += self.detector.update(_state(3, 1.5, False))
        self.assertEqual(len(events), 0)

    def test_end_flicker_within_debounce_suppressed(self) -> None:
        events: list = []
        for i in range(3):
            events += self.detector.update(_state(i, i * 0.5, True))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")

        events += self.detector.update(_state(3, 1.5, False))
        events += self.detector.update(_state(4, 2.0, True))
        self.assertEqual(len(events), 1, "Flicker should not emit ROUND_END")

    def test_exactly_debounce_frames_needed(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=3)
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertEqual(len(events), 0, "2 frames insufficient for debounce=3")
        events += detector.update(_state(2, 1.0, True))
        self.assertEqual(len(events), 1, "3rd frame should confirm")


class RoundBoundaryDetectorMinDurationTests(TestCase):
    """Tests for minimum round duration enforcement."""

    def test_short_round_suppressed(self) -> None:
        detector = RoundBoundaryDetector(
            debounce_frames=2, min_round_duration_sec=10.0
        )
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")

        events += detector.update(_state(2, 1.0, False))
        events += detector.update(_state(3, 1.5, False))
        self.assertEqual(len(events), 1, "Short round should not produce ROUND_END")
        self.assertEqual(detector._phase, "in_round")

    def test_round_longer_than_minimum_not_suppressed(self) -> None:
        detector = RoundBoundaryDetector(
            debounce_frames=2, min_round_duration_sec=3.0
        )
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertEqual(len(events), 1)

        events += detector.update(_state(10, 5.0, False))
        events += detector.update(_state(11, 5.5, False))
        self.assertEqual(len(events), 2)
        _assert_event(self, events[1], EventType.ROUND_END, "round_001")

    def test_min_duration_zero_allows_all_rounds(self) -> None:
        detector = RoundBoundaryDetector(
            debounce_frames=2, min_round_duration_sec=0.0
        )
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        events += detector.update(_state(2, 1.0, False))
        events += detector.update(_state(3, 1.5, False))
        self.assertEqual(len(events), 2)


class RoundBoundaryDetectorFinalizeTests(TestCase):
    """Tests for finalize() behaviour."""

    def test_finalize_mid_round_emits_round_end(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2)
        detector.update(_state(0, 0.0, True))
        detector.update(_state(1, 0.5, True))
        final = detector.finalize()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].event_type, EventType.ROUND_END)

    def test_finalize_candidate_start_emits_round_end(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=5)
        detector.update(_state(0, 0.0, True))
        final = detector.finalize()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].event_type, EventType.ROUND_END)

    def test_finalize_idle_does_nothing(self) -> None:
        detector = RoundBoundaryDetector()
        detector.update(_state(0, 0.0, False))
        final = detector.finalize()
        self.assertEqual(len(final), 0)

    def test_finalize_resets_state_for_reuse(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2)
        detector.update(_state(0, 0.0, True))
        detector.update(_state(1, 0.5, True))
        detector.finalize()

        self.assertEqual(detector._phase, "idle")
        self.assertEqual(detector._round_counter, 0)

        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")


class RoundBoundaryDetectorMultiRoundTests(TestCase):
    """Tests for multiple consecutive rounds."""

    def test_two_complete_rounds(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2, min_round_duration_sec=0)

        def feed(fi: int, ts: float, active: bool):
            return list(detector.update(_state(fi, ts, active)))

        events: list = []
        # Round 1
        events += feed(0, 0.0, True)
        events += feed(1, 0.5, True)
        for i in range(2, 20):
            events += feed(i, i * 0.5, True)
        events += feed(20, 10.0, False)
        events += feed(21, 10.5, False)

        # Round 2
        events += feed(40, 20.0, True)
        events += feed(41, 20.5, True)
        for i in range(42, 80):
            events += feed(i, i * 0.5, True)
        events += feed(80, 40.0, False)
        events += feed(81, 40.5, False)

        starts = [e for e in events if e.event_type == EventType.ROUND_START]
        ends = [e for e in events if e.event_type == EventType.ROUND_END]
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(ends), 2)
        _assert_event(self, starts[0], EventType.ROUND_START, "round_001")
        _assert_event(self, ends[0], EventType.ROUND_END, "round_001")
        _assert_event(self, starts[1], EventType.ROUND_START, "round_002")
        _assert_event(self, ends[1], EventType.ROUND_END, "round_002")

    def test_round_counter_increments(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=1, min_round_duration_sec=0)
        events: list = []

        # Round 1: start is emitted immediately with debounce=1
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 1.0, False))
        self.assertEqual(len(events), 2)

        # Round 2
        events += detector.update(_state(2, 2.0, True))
        events += detector.update(_state(3, 3.0, False))
        self.assertEqual(len(events), 4)

        _assert_event(self, events[2], EventType.ROUND_START, "round_002")
        _assert_event(self, events[3], EventType.ROUND_END, "round_002")


class RoundBoundaryDetectorCustomKeyTests(TestCase):
    """Tests with a custom state_key."""

    def test_custom_key_works(self) -> None:
        detector = RoundBoundaryDetector(
            state_key="custom.round_flag",
            debounce_frames=2,
            min_round_duration_sec=0,
        )
        events: list = []
        events += detector.update(_state(0, 0.0, True, key="custom.round_flag"))
        events += detector.update(_state(1, 0.5, True, key="custom.round_flag"))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")

    def test_missing_key_treated_as_false(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2)
        state = HudState(
            frame_index=0, timestamp_sec=0.0,
            profile="cs2", values={}, confidence=0.5,
        )
        events = list(detector.update(state))
        self.assertEqual(len(events), 0)


class RoundBoundaryDetectorEvidenceTests(TestCase):
    """Tests for evidence and event metadata."""

    def test_event_has_evidence(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2, min_round_duration_sec=0)
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertGreater(len(events[0].evidence), 0)
        self.assertEqual(
            events[0].evidence[0].source,
            "RoundBoundaryDetector.round_info.round_active",
        )

    def test_event_has_round_id_in_attributes(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2, min_round_duration_sec=0)
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertEqual(events[0].attributes["round_id"], "round_001")

    def test_event_confidence_is_high(self) -> None:
        detector = RoundBoundaryDetector(debounce_frames=2, min_round_duration_sec=0)
        events: list = []
        events += detector.update(_state(0, 0.0, True))
        events += detector.update(_state(1, 0.5, True))
        self.assertGreaterEqual(events[0].confidence, 0.8)

# -- KillEventDetector -------------------------------------------------------

def _hud_state(
    fi: int, ts: float,
    hp: int | None = 100,
    kill_feed: bool = False,
) -> HudState:
    """Factory for a HudState with HP and kill-feed values."""
    return HudState(
        frame_index=fi,
        timestamp_sec=ts,
        profile="cs2_standard_16x9",
        values={
            "player_status.hp": hp,
            "kill_feed.kill_feed_active": kill_feed,
        },
        confidence=0.8,
    )


class KillEventDetectorInterfaceTests(TestCase):
    """Contract and constructor tests."""

    def test_implements_event_engine(self) -> None:
        detector = KillEventDetector()
        self.assertIsInstance(detector, EventEngine)

    def test_default_constructor_values(self) -> None:
        detector = KillEventDetector()
        self.assertEqual(detector._hp_key, "player_status.hp")
        self.assertEqual(detector._kf_key, "kill_feed.kill_feed_active")
        self.assertEqual(detector._hp_death_threshold, 20)
        self.assertEqual(detector._death_debounce, 4)
        self.assertEqual(detector._kill_debounce, 2)

    def test_raises_on_invalid_death_debounce(self) -> None:
        with self.assertRaises(ValueError):
            KillEventDetector(death_debounce_frames=0)

    def test_raises_on_invalid_kill_debounce(self) -> None:
        with self.assertRaises(ValueError):
            KillEventDetector(kill_debounce_frames=0)

    def test_finalize_returns_empty(self) -> None:
        detector = KillEventDetector()
        self.assertEqual(len(detector.finalize()), 0)


class KillEventDetectorDeathTests(TestCase):
    """Tests for PLAYER_DEATH detection."""

    def test_sustained_low_hp_emits_death(self) -> None:
        detector = KillEventDetector(
            death_debounce_frames=3,
            death_cooldown_sec=0,
        )
        events: list = []

        # Player alive at high HP
        for i in range(3):
            events += detector.update(_hud_state(i, i * 0.5, hp=100))

        # HP drops below threshold 鈥?need 3 consecutive frames
        events += detector.update(_hud_state(3, 1.5, hp=10))
        events += detector.update(_hud_state(4, 2.0, hp=5))
        self.assertEqual(len(events), 0, "2 low-HP frames insufficient for debounce=3")

        events += detector.update(_hud_state(5, 2.5, hp=3))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.PLAYER_DEATH)

    def test_hp_at_threshold_does_not_trigger(self) -> None:
        detector = KillEventDetector(
            hp_death_threshold=20, death_debounce_frames=1, death_cooldown_sec=0,
        )
        events: list = []
        events += detector.update(_hud_state(0, 0.0, hp=20))  # exactly at threshold
        self.assertEqual(len(events), 0, "HP at threshold should not trigger death")

    def test_hp_below_threshold_triggers(self) -> None:
        detector = KillEventDetector(
            hp_death_threshold=20, death_debounce_frames=1, death_cooldown_sec=0,
        )
        events = list(detector.update(_hud_state(0, 0.0, hp=19)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.PLAYER_DEATH)

    def test_hp_recovery_resets_debounce(self) -> None:
        detector = KillEventDetector(
            death_debounce_frames=4, death_cooldown_sec=0,
        )
        events: list = []

        events += detector.update(_hud_state(0, 0.0, hp=10))
        events += detector.update(_hud_state(1, 0.5, hp=8))
        # Recovery 鈥?HP goes back up
        events += detector.update(_hud_state(2, 1.0, hp=80))
        self.assertEqual(len(events), 0)

        # HP drops again 鈥?debounce should restart
        events += detector.update(_hud_state(3, 1.5, hp=5))
        events += detector.update(_hud_state(4, 2.0, hp=3))
        events += detector.update(_hud_state(5, 2.5, hp=2))
        events += detector.update(_hud_state(6, 3.0, hp=1))
        self.assertEqual(len(events), 1, "Debounce should restart after recovery")

    def test_death_cooldown_prevents_duplicates(self) -> None:
        detector = KillEventDetector(
            death_debounce_frames=1,
            death_cooldown_sec=5.0,
        )
        events: list = []

        # First death
        events += detector.update(_hud_state(0, 0.0, hp=10))
        self.assertEqual(len(events), 1)

        # Immediately low HP again 鈥?cooldown suppresses
        events += detector.update(_hud_state(1, 0.5, hp=5))
        self.assertEqual(len(events), 1)

        # After cooldown 鈥?new death possible
        events += detector.update(_hud_state(20, 5.5, hp=3))
        self.assertEqual(len(events), 2)

    def test_missing_hp_key_does_not_crash(self) -> None:
        detector = KillEventDetector(death_debounce_frames=1, death_cooldown_sec=0)
        state = HudState(
            frame_index=0, timestamp_sec=0.0,
            profile="cs2", values={}, confidence=0.5,
        )
        events = list(detector.update(state))
        self.assertEqual(len(events), 0)

    def test_death_event_has_correct_metadata(self) -> None:
        detector = KillEventDetector(
            death_debounce_frames=1, death_cooldown_sec=0,
        )
        events = list(detector.update(_hud_state(5, 3.0, hp=5)))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e.event_id, "player_death_001")
        self.assertGreaterEqual(e.confidence, 0.8)
        self.assertEqual(e.attributes["death_index"], 1)
        self.assertGreater(len(e.evidence), 0)


class KillEventDetectorKillTests(TestCase):
    """Tests for PLAYER_KILL detection via kill-feed rising edges."""

    def test_rising_edge_emits_kill(self) -> None:
        detector = KillEventDetector(
            kill_debounce_frames=2, kill_cooldown_sec=0,
        )
        events: list = []

        # Kill feed inactive
        events += detector.update(_hud_state(0, 0.0, kill_feed=False))
        events += detector.update(_hud_state(1, 0.5, kill_feed=False))

        # Rising edge 鈥?need 2 frames
        events += detector.update(_hud_state(2, 1.0, kill_feed=True))
        self.assertEqual(len(events), 0, "1 frame of rising edge insufficient")
        events += detector.update(_hud_state(3, 1.5, kill_feed=True))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.PLAYER_KILL)

    def test_kill_feed_staying_active_no_extra_events(self) -> None:
        detector = KillEventDetector(
            kill_debounce_frames=1, kill_cooldown_sec=0,
        )
        events: list = []

        # First rising edge 鈫?kill
        events += detector.update(_hud_state(0, 0.0, kill_feed=True))
        self.assertEqual(len(events), 1)

        # Staying active 鈥?no new events
        for i in range(1, 10):
            events += detector.update(_hud_state(i, i * 0.5, kill_feed=True))
        kill_count = sum(1 for e in events if e.event_type == EventType.PLAYER_KILL)
        self.assertEqual(kill_count, 1)

    def test_kill_cooldown_prevents_duplicates(self) -> None:
        detector = KillEventDetector(
            kill_debounce_frames=1,
            kill_cooldown_sec=3.0,
        )
        events: list = []

        # First kill
        events += detector.update(_hud_state(0, 0.0, kill_feed=True))
        self.assertEqual(len(events), 1)

        # Feed goes inactive, then active again within cooldown
        events += detector.update(_hud_state(1, 0.5, kill_feed=False))
        events += detector.update(_hud_state(2, 1.0, kill_feed=True))
        self.assertEqual(len(events), 1, "Cooldown should suppress rapid re-trigger")

        # After cooldown
        events += detector.update(_hud_state(10, 3.5, kill_feed=False))
        events += detector.update(_hud_state(11, 4.0, kill_feed=True))
        self.assertEqual(len(events), 2)

    def test_kill_event_has_metadata(self) -> None:
        detector = KillEventDetector(
            kill_debounce_frames=1, kill_cooldown_sec=0,
        )
        events = list(detector.update(_hud_state(0, 0.0, kill_feed=True)))
        e = events[0]
        self.assertEqual(e.event_id, "player_kill_001")
        self.assertEqual(e.event_type, EventType.PLAYER_KILL)
        self.assertLess(e.confidence, 0.8,
                        msg="Kill confidence should be lower than death (colour-only heuristic)")
        self.assertEqual(e.attributes["kill_index"], 1)

    def test_multiple_kills_across_round(self) -> None:
        detector = KillEventDetector(
            kill_debounce_frames=1, kill_cooldown_sec=1.0,
        )
        events: list = []

        # Kill 1
        events += detector.update(_hud_state(0, 0.0, kill_feed=True))
        events += detector.update(_hud_state(1, 0.5, kill_feed=False))

        # Wait for cooldown
        events += detector.update(_hud_state(5, 1.5, kill_feed=False))

        # Kill 2
        events += detector.update(_hud_state(6, 2.0, kill_feed=True))
        events += detector.update(_hud_state(7, 2.5, kill_feed=False))

        # Wait for cooldown
        events += detector.update(_hud_state(12, 3.5, kill_feed=False))

        # Kill 3
        events += detector.update(_hud_state(13, 4.0, kill_feed=True))

        kills = [e for e in events if e.event_type == EventType.PLAYER_KILL]
        self.assertEqual(len(kills), 3)
        self.assertEqual(kills[0].event_id, "player_kill_001")
        self.assertEqual(kills[1].event_id, "player_kill_002")
        self.assertEqual(kills[2].event_id, "player_kill_003")


class KillEventDetectorCombinedTests(TestCase):
    """Tests where both death and kill signals fire together."""

    def test_death_and_kill_in_same_frame(self) -> None:
        detector = KillEventDetector(
            death_debounce_frames=1, death_cooldown_sec=0,
            kill_debounce_frames=1, kill_cooldown_sec=0,
        )
        # Both HP low and kill feed active
        state = HudState(
            frame_index=0, timestamp_sec=0.0,
            profile="cs2", values={
                "player_status.hp": 5,
                "kill_feed.kill_feed_active": True,
            }, confidence=0.8,
        )
        events = list(detector.update(state))
        types = {e.event_type for e in events}
        self.assertIn(EventType.PLAYER_DEATH, types)
        self.assertIn(EventType.PLAYER_KILL, types)

    def test_death_cooldown_does_not_block_kills(self) -> None:
        """Death cooldown should only affect death events, not kills."""
        detector = KillEventDetector(
            death_debounce_frames=1, death_cooldown_sec=10.0,
            kill_debounce_frames=1, kill_cooldown_sec=0,
        )
        events: list = []

        # Death at t=0
        events += detector.update(_hud_state(0, 0.0, hp=3))

        # Kill feed activates during death cooldown 鈥?should still emit
        events += detector.update(_hud_state(1, 0.5, kill_feed=True))

        types_after = {e.event_type for e in events}
        self.assertIn(EventType.PLAYER_KILL, types_after)

    def test_finalize_resets_counters(self) -> None:
        detector = KillEventDetector(
            death_debounce_frames=1, death_cooldown_sec=0,
            kill_debounce_frames=1, kill_cooldown_sec=0,
        )
        detector.update(_hud_state(0, 0.0, hp=5))
        detector.update(_hud_state(1, 0.5, kill_feed=True))
        detector.finalize()

        # Reuse 鈥?counters should start fresh
        self.assertEqual(detector._death_counter, 0)
        self.assertEqual(detector._kill_counter, 0)
