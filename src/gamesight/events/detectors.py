"""Event detectors that consume HudState sequences and emit GameEvent objects.

Each detector implements ``EventEngine`` so it can be composed into a
streaming pipeline or used standalone with a list of states.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from gamesight.domain.models import EventType, Evidence, GameEvent, HudState, Track
from gamesight.events.engine import EventEngine


_SMOOTH_WINDOW = 5
_RATIO_LOW = 0.001
_RATIO_HIGH = 0.006
_SCORE_CHANGE_RATIO = 0.30  # 30% pixel count change = score changed


class RoundBoundaryDetector(EventEngine):
    """Detect round boundaries with dual confirmation.

    A new round requires BOTH:
      1. Timer reappears (smoothed pixel ratio > threshold after being absent)
      2. Score numbers changed (CT blue or T yellow pixel count shifted >30%)

    This prevents false triggers during freeze-time countdown colour changes
    (white -> red) and bomb-plant timer transitions.

    Parameters
    ----------
    ratio_key:
        HudState key for timer_pixel_ratio.
    ct_pixels_key:
        HudState key for CT score blue pixel count.
    t_pixels_key:
        HudState key for T score yellow pixel count.
    smooth_window:
        Frames for rolling average of timer ratio.
    ratio_high / ratio_low:
        Hysteresis thresholds for timer visibility.
    score_change_ratio:
        Minimum fraction change in score pixels to flag a score update.
    min_round_duration_sec:
        Suppress implausibly short rounds.
    """

    def __init__(
        self,
        ratio_key: str = "round_info.timer_pixel_ratio",
        ct_pixels_key: str = "round_info.ct_score_pixels",
        t_pixels_key: str = "round_info.t_score_pixels",
        smooth_window: int = _SMOOTH_WINDOW,
        ratio_high: float = _RATIO_HIGH,
        ratio_low: float = _RATIO_LOW,
        score_change_ratio: float = _SCORE_CHANGE_RATIO,
        min_round_duration_sec: float = 3.0,
    ) -> None:
        self._ratio_key = ratio_key
        self._ct_key = ct_pixels_key
        self._t_key = t_pixels_key
        self._smooth_win = max(1, smooth_window)
        self._ratio_high = ratio_high
        self._ratio_low = ratio_low
        self._score_change = score_change_ratio
        self._min_duration = min_round_duration_sec

        self._history: deque[float] = deque(maxlen=self._smooth_win)
        self._smoothed_ratio = 0.0
        self._round_counter = 0
        self._in_round = False
        self._last_start_ts: float | None = None
        self._absence_count = 0
        self._presence_count = 0
        # Score tracking: pixel counts from the last confirmed round.
        self._last_ct_px: int = -1
        self._last_t_px: int = -1
        self._pending_events: list[GameEvent] = []

    # -- EventEngine interface -----------------------------------------------

    def update(
        self, hud_state: HudState, tracks: Sequence[Track] = ()
    ) -> Sequence[GameEvent]:
        self._pending_events.clear()

        raw = float(hud_state.values.get(self._ratio_key, 0))
        ct_px = int(hud_state.values.get(self._ct_key, 0))
        t_px = int(hud_state.values.get(self._t_key, 0))
        fi = hud_state.frame_index
        ts = hud_state.timestamp_sec

        self._history.append(raw)
        self._smoothed_ratio = sum(self._history) / len(self._history)

        timer_present = self._smoothed_ratio > self._ratio_high
        timer_absent = self._smoothed_ratio < self._ratio_low

        if not self._in_round:
            if timer_present:
                self._presence_count += 1
                if self._presence_count >= 5:
                    # Dual confirm: timer present AND scores changed.
                    if self._scores_changed(ct_px, t_px):
                        self._confirm_round_start(fi, ts, ct_px, t_px)
                    # Allow first round without score change.
                    elif self._last_ct_px < 0:
                        self._confirm_round_start(fi, ts, ct_px, t_px)
            else:
                self._presence_count = 0
        else:
            if timer_absent:
                self._absence_count += 1
                if self._absence_count >= 10:
                    self._confirm_round_end(fi, ts)
            else:
                self._absence_count = max(0, self._absence_count - 1)
                # Track scores during round for change detection.
                if ct_px > 0:
                    self._last_ct_px = ct_px
                if t_px > 0:
                    self._last_t_px = t_px

        return tuple(self._pending_events)

    def finalize(self) -> Sequence[GameEvent]:
        events: list[GameEvent] = []
        if self._in_round:
            self._round_counter += 1
            rid = f"round_{self._round_counter:03d}"
            events.append(self._make_event(EventType.ROUND_END, rid, 0.0, fi=None))
        self._reset()
        return events

    # -- internal ------------------------------------------------------------

    def _scores_changed(self, ct_px: int, t_px: int) -> bool:
        """Return True if either CT or T score pixel count changed significantly."""
        if self._last_ct_px < 0 or self._last_t_px < 0:
            return False
        ct_changed = (
            abs(ct_px - self._last_ct_px) / max(self._last_ct_px, 1)
            > self._score_change
        )
        t_changed = (
            abs(t_px - self._last_t_px) / max(self._last_t_px, 1)
            > self._score_change
        )
        return ct_changed or t_changed

    def _confirm_round_start(
        self, fi: int, ts: float, ct_px: int, t_px: int
    ) -> None:
        self._round_counter += 1
        rid = f"round_{self._round_counter:03d}"
        self._in_round = True
        self._last_start_ts = ts
        self._absence_count = 0
        self._presence_count = 0
        self._pending_events.append(
            self._make_event(EventType.ROUND_START, rid, ts, fi=fi)
        )

    def _confirm_round_end(self, fi: int, ts: float) -> None:
        rid = f"round_{self._round_counter:03d}"
        if (
            self._last_start_ts is not None
            and (ts - self._last_start_ts) < self._min_duration
        ):
            self._absence_count = 0
            return
        self._in_round = False
        self._absence_count = 0
        self._presence_count = 0
        self._pending_events.append(
            self._make_event(EventType.ROUND_END, rid, ts, fi=fi)
        )

    def _make_event(
        self, event_type: EventType, round_id: str, ts: float, fi: int | None
    ) -> GameEvent:
        return GameEvent(
            event_id=f"{event_type.value}_{round_id}",
            event_type=event_type,
            start_sec=ts,
            confidence=0.9,
            evidence=[
                Evidence(
                    frame_index=fi,
                    timestamp_sec=ts,
                    source=f"RoundBoundaryDetector.{self._ratio_key}",
                )
            ],
            attributes={"round_id": round_id},
        )

    def _reset(self) -> None:
        self._history.clear()
        self._smoothed_ratio = 0.0
        self._round_counter = 0
        self._in_round = False
        self._last_start_ts = None
        self._absence_count = 0
        self._presence_count = 0
        self._last_ct_px = -1
        self._last_t_px = -1
        self._pending_events.clear()


class KillEventDetector(EventEngine):
    """Detect PLAYER_DEATH and PLAYER_KILL from HP drops and kill-feed activity.

    Death detection watches ``player_status.hp`` for a sustained drop below
    a configurable threshold.  Kill detection watches ``kill_feed.kill_feed_active``
    for rising edges (new entries appearing in the feed).

    Both detectors use debounce windows to suppress noise and cooldown
    periods to prevent duplicate events for the same in-game kill or death.
    """

    def __init__(
        self,
        hp_key: str = "player_status.hp",
        kill_feed_key: str = "kill_feed.kill_feed_active",
        hp_death_threshold: int = 20,
        death_debounce_frames: int = 4,
        kill_debounce_frames: int = 2,
        death_cooldown_sec: float = 8.0,
        kill_cooldown_sec: float = 2.0,
    ) -> None:
        if death_debounce_frames < 1:
            raise ValueError("death_debounce_frames must be >= 1")
        if kill_debounce_frames < 1:
            raise ValueError("kill_debounce_frames must be >= 1")

        self._hp_key = hp_key
        self._kf_key = kill_feed_key
        self._hp_death_threshold = hp_death_threshold
        self._death_debounce = death_debounce_frames
        self._kill_debounce = kill_debounce_frames
        self._death_cooldown = death_cooldown_sec
        self._kill_cooldown = kill_cooldown_sec

        self._kill_counter = 0
        self._death_counter = 0
        self._death_debounce_count = 0
        self._kill_feed_was_active = False
        self._kill_rising_count = 0
        self._last_death_ts: float | None = None
        self._last_kill_ts: float | None = None
        self._pending_events: list[GameEvent] = []

    def update(
        self, hud_state: HudState, tracks: Sequence[Track] = ()
    ) -> Sequence[GameEvent]:
        self._pending_events.clear()
        ts = hud_state.timestamp_sec
        fi = hud_state.frame_index
        self._detect_death(hud_state, fi, ts)
        self._detect_kill(hud_state, fi, ts)
        return tuple(self._pending_events)

    def finalize(self) -> Sequence[GameEvent]:
        self._reset()
        return ()

    def _detect_death(self, state: HudState, fi: int, ts: float) -> None:
        hp = state.values.get(self._hp_key)
        if not isinstance(hp, (int, float)) or hp is None:
            return
        in_cooldown = (
            self._last_death_ts is not None
            and (ts - self._last_death_ts) < self._death_cooldown
        )
        if hp < self._hp_death_threshold and not in_cooldown:
            self._death_debounce_count += 1
            if self._death_debounce_count >= self._death_debounce:
                self._emit_death(fi, ts)
        else:
            self._death_debounce_count = max(0, self._death_debounce_count - 1)

    def _emit_death(self, fi: int, ts: float) -> None:
        self._death_counter += 1
        self._last_death_ts = ts
        self._death_debounce_count = 0
        self._pending_events.append(GameEvent(
            event_id=f"player_death_{self._death_counter:03d}",
            event_type=EventType.PLAYER_DEATH, start_sec=ts, confidence=0.85,
            evidence=[Evidence(frame_index=fi, timestamp_sec=ts, source=f"KillEventDetector.{self._hp_key}")],
            attributes={"death_index": self._death_counter, "hp_key": self._hp_key},
        ))

    def _detect_kill(self, state: HudState, fi: int, ts: float) -> None:
        current = bool(state.values.get(self._kf_key, False))
        in_cooldown = (
            self._last_kill_ts is not None
            and (ts - self._last_kill_ts) < self._kill_cooldown
        )
        if current and not self._kill_feed_was_active and not in_cooldown:
            self._kill_rising_count = 1
        elif current and self._kill_rising_count > 0:
            self._kill_rising_count += 1
        else:
            self._kill_rising_count = 0
        if self._kill_rising_count >= self._kill_debounce:
            self._emit_kill(fi, ts)
        self._kill_feed_was_active = current

    def _emit_kill(self, fi: int, ts: float) -> None:
        self._kill_counter += 1
        self._last_kill_ts = ts
        self._kill_rising_count = 0
        self._pending_events.append(GameEvent(
            event_id=f"player_kill_{self._kill_counter:03d}",
            event_type=EventType.PLAYER_KILL, start_sec=ts, confidence=0.55,
            evidence=[Evidence(frame_index=fi, timestamp_sec=ts, source=f"KillEventDetector.{self._kf_key}")],
            attributes={"kill_index": self._kill_counter, "kf_key": self._kf_key},
        ))

    def _reset(self) -> None:
        self._kill_counter = self._death_counter = 0
        self._death_debounce_count = 0
        self._kill_feed_was_active = False
        self._kill_rising_count = 0
        self._last_death_ts = self._last_kill_ts = None
        self._pending_events.clear()