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



def _ratio_state(
    fi: int, ts: float, ratio: float,
    ct_px: int = 0, t_px: int = 0,
):
    from gamesight.domain.models import HudState
    return HudState(
        frame_index=fi, timestamp_sec=ts,
        values={
            "round_info.timer_pixel_ratio": ratio,
            "round_info.ct_score_pixels": ct_px,
            "round_info.t_score_pixels": t_px,
        },
        confidence=0.5,
    )


def _hp_state(fi: int, ts: float, hp: int, kf: bool = False):
    from gamesight.domain.models import HudState
    return HudState(
        frame_index=fi, timestamp_sec=ts,
        values={"player_status.hp": hp, "kill_feed.kill_feed_active": kf}, confidence=0.5,
    )


def _hud_state(fi: int = 0, ts: float = 0.0, hp: int = 100, kill_feed: bool = False):
    from gamesight.domain.models import HudState
    return HudState(
        frame_index=fi, timestamp_sec=ts,
        values={
            "player_status.hp": hp,
            "kill_feed.kill_feed_active": kill_feed,
        },
        confidence=0.5,
    )


def _assert_event(tc, event, event_type, round_id):
    from gamesight.domain.models import EventType
    tc.assertEqual(event.event_type, event_type)
    tc.assertEqual(event.attributes.get("round_id"), round_id)


class RoundBoundaryDetectorInterfaceTests(TestCase):

    def test_implements_event_engine(self) -> None:
        from gamesight.events.engine import EventEngine
        detector = RoundBoundaryDetector()
        self.assertIsInstance(detector, EventEngine)

    def test_default_constructor_values(self) -> None:
        detector = RoundBoundaryDetector()
        self.assertEqual(detector._ratio_key, "round_info.timer_pixel_ratio")
        self.assertEqual(detector._ct_key, "round_info.ct_score_pixels")
        self.assertEqual(detector._t_key, "round_info.t_score_pixels")
        self.assertEqual(detector._smooth_win, 1)
        self.assertEqual(detector._ratio_high, 0.006)
        self.assertEqual(detector._ratio_low, 0.001)
        self.assertEqual(detector._min_duration, 15.0)
        self.assertEqual(detector._presence_confirm, 2.0)
        self.assertEqual(detector._absence_confirm, 2.0)

    def test_update_returns_sequence(self) -> None:
        detector = RoundBoundaryDetector()
        result = detector.update(_ratio_state(0, 0.0, 0.0))
        self.assertIsInstance(result, tuple)

    def test_custom_ratio_key(self) -> None:
        detector = RoundBoundaryDetector(ratio_key="custom.round_active")
        self.assertEqual(detector._ratio_key, "custom.round_active")

    def test_custom_smooth_window(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=10)
        self.assertEqual(detector._smooth_win, 10)


class RoundBoundaryDetectorBasicTests(TestCase):

    def test_complete_round_lifecycle(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")
        for i in range(5, 100):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        # Timer absent + scores change -> round ends.
        for i in range(100, 116):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=200, t_px=150))
        self.assertEqual(len(events), 2)
        _assert_event(self, events[1], EventType.ROUND_END, "round_001")

    def test_two_complete_rounds(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        events: list = []
        # Round 1: first round, no prior scores needed.
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")
        for i in range(5, 110):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        for i in range(110, 126):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=0, t_px=0))
        self.assertEqual(len(events), 2)
        _assert_event(self, events[1], EventType.ROUND_END, "round_001")
        # Round 2: scores changed (200 vs 100), timer reappears -> new round.
        for i in range(126, 131):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=200, t_px=150))
        self.assertEqual(len(events), 3)
        _assert_event(self, events[2], EventType.ROUND_START, "round_002")
        for i in range(131, 235):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=200, t_px=150))
        for i in range(235, 251):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=0, t_px=0))
        self.assertEqual(len(events), 4)
        _assert_event(self, events[3], EventType.ROUND_END, "round_002")

    def test_round_start_requires_5_presence_frames(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        events: list = []
        for i in range(3):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01))
        self.assertEqual(len(events), 0)
        for i in range(3, 5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01))
        self.assertEqual(len(events), 1)

    def test_round_end_requires_absence_and_score_change(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        events: list = []
        # Round 1 start (first round exemption, no prior scores).
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        # Timer absent for 15 frames but scores unchanged -> no round end (C4 planted scenario).
        for i in range(5, 25):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1, "Timer absent, scores same -> no end")
        # Scores change -> round ends.
        for i in range(25, 41):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=200, t_px=150))
        self.assertEqual(len(events), 2)
        _assert_event(self, events[1], EventType.ROUND_END, "round_001")


class RoundBoundaryDetectorMinDurationTests(TestCase):

    def test_short_round_suppressed(self) -> None:
        detector = RoundBoundaryDetector(
            smooth_window=1, ratio_high=0.003, ratio_low=0.001,
            min_round_duration_sec=5.0, presence_confirm_sec=0.4,
            absence_confirm_sec=1.5,
        )
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.1, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        # Scores change but round too short.
        for i in range(5, 25):
            events += detector.update(_ratio_state(i, i * 0.1, 0.0, ct_px=200, t_px=150))
        self.assertEqual(len(events), 1, "Short round suppressed")

    def test_long_round_not_suppressed(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001, min_round_duration_sec=1.0)
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        for i in range(5, 20):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        for i in range(20, 36):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=200, t_px=150))
        self.assertEqual(len(events), 2)

    def test_min_duration_zero_allows_all(self) -> None:
        detector = RoundBoundaryDetector(
            smooth_window=1, ratio_high=0.003, ratio_low=0.001,
            min_round_duration_sec=0.0, presence_confirm_sec=0.4,
            absence_confirm_sec=1.5,
        )
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.1, 0.01, ct_px=100, t_px=100))
        for i in range(5, 21):
            events += detector.update(_ratio_state(i, i * 0.1, 0.0, ct_px=200, t_px=150))
        self.assertEqual(len(events), 2)


class RoundBoundaryDetectorSmoothingTests(TestCase):

    def test_smoothing_filters_flickers(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=5, ratio_high=0.005, ratio_low=0.001)
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01))
        self.assertEqual(len(events), 1)
        for i in range(5, 7):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0))
        self.assertEqual(len(events), 1)
        for i in range(7, 20):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01))
        self.assertEqual(len(events), 1, "Flicker filtered by smoothing")

    def test_sustained_absence_with_score_change_ends_round(self) -> None:
        detector = RoundBoundaryDetector(
            smooth_window=1, ratio_high=0.003, ratio_low=0.001,
            min_round_duration_sec=1.0,
        )
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        self.assertEqual(len(events), 1)
        for i in range(5, 21):
            events += detector.update(_ratio_state(i, i * 0.5, 0.0, ct_px=200, t_px=150))
        self.assertEqual(len(events), 2)


class RoundBoundaryDetectorTimerGapFallbackTests(TestCase):

    def _run_at_fps(self, sample_fps: int) -> list:
        detector = RoundBoundaryDetector(
            smooth_window=1,
            ratio_high=0.003,
            ratio_low=0.001,
            min_round_duration_sec=15.0,
            presence_confirm_sec=2.0,
            absence_confirm_sec=2.0,
        )
        events: list = []
        fi = 0
        step = 1.0 / sample_fps
        # Recognised timer, then a long C4/end gap, then the next timer.
        # Score-colour pixels remain unavailable throughout.
        for start, end, ratio in ((0.0, 40.0, 0.01), (40.0, 55.0, 0.0), (55.0, 75.0, 0.01)):
            ts = start
            while ts < end:
                events += detector.update(_ratio_state(fi, ts, ratio, ct_px=20, t_px=0))
                fi += 1
                ts += step
        return events

    def test_custom_hud_without_score_colours_detects_next_round(self) -> None:
        events = self._run_at_fps(2)
        self.assertEqual([e.event_type for e in events], [
            EventType.ROUND_START,
            EventType.ROUND_END,
            EventType.ROUND_START,
        ])
        self.assertEqual(events[0].attributes["round_id"], "round_001")
        self.assertEqual(events[2].attributes["round_id"], "round_002")
        self.assertAlmostEqual(events[1].start_sec, 55.0, places=1)
        self.assertAlmostEqual(events[2].start_sec, 55.0, places=1)

    def test_boundaries_are_sampling_rate_independent(self) -> None:
        at_2_fps = self._run_at_fps(2)
        at_10_fps = self._run_at_fps(10)
        self.assertEqual(len(at_2_fps), len(at_10_fps))
        for low_rate, high_rate in zip(at_2_fps, at_10_fps):
            self.assertEqual(low_rate.event_type, high_rate.event_type)
            self.assertAlmostEqual(low_rate.start_sec, high_rate.start_sec, places=1)

    def test_mid_range_ratio_ignored(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.005, ratio_low=0.001)
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01))
        self.assertEqual(len(events), 1)
        for i in range(5, 30):
            events += detector.update(_ratio_state(i, i * 0.5, 0.003))
        self.assertEqual(len(events), 1)


class RoundBoundaryDetectorFinalizeTests(TestCase):

    def test_finalize_mid_round_emits_round_end(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        for i in range(5):
            detector.update(_ratio_state(i, i * 0.5, 0.01, ct_px=100, t_px=100))
        final = detector.finalize()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].event_type, EventType.ROUND_END)

    def test_finalize_idle_does_nothing(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        final = detector.finalize()
        self.assertEqual(len(final), 0)

    def test_finalize_resets_state(self) -> None:
        detector = RoundBoundaryDetector(smooth_window=1, ratio_high=0.003, ratio_low=0.001)
        for i in range(5):
            detector.update(_ratio_state(i, i * 0.5, 0.01))
        detector.finalize()
        events: list = []
        for i in range(5):
            events += detector.update(_ratio_state(i, i * 0.5, 0.01))
        self.assertEqual(len(events), 1)
        _assert_event(self, events[0], EventType.ROUND_START, "round_001")

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
