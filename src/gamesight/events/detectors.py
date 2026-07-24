"""Event detectors that consume HudState sequences and emit GameEvent objects.

Each detector implements ``EventEngine`` so it can be composed into a
streaming pipeline or used standalone with a list of states.
"""

from __future__ import annotations

from collections.abc import Sequence

from gamesight.domain.models import EventType, Evidence, GameEvent, HudState, Track
from gamesight.events.engine import EventEngine


class RoundBoundaryDetector(EventEngine):
    """Detect round start / end by tracking ``round_active`` state transitions.

    Uses a configurable debounce window so that brief flickers in the
    colour-heuristic HUD extraction do not produce false events.  A
    minimum round duration rejects implausibly short rounds (e.g. caused
    by HUD extraction noise during freeze time).

    Parameters
    ----------
    state_key:
        The ``HudState.values`` key that carries the ``round_active``
        boolean from the ``RoundInfoExtractor``.
    debounce_frames:
        Number of *consecutive* frames that must agree before a
        transition is confirmed.
    min_round_duration_sec:
        Rounds shorter than this are suppressed (start+end pair dropped).
    """

    def __init__(
        self,
        state_key: str = "round_info.round_active",
        debounce_frames: int = 3,
        min_round_duration_sec: float = 5.0,
    ) -> None:
        if debounce_frames < 1:
            raise ValueError("debounce_frames must be >= 1")
        if min_round_duration_sec < 0:
            raise ValueError("min_round_duration_sec must be >= 0")

        self._state_key = state_key
        self._debounce = debounce_frames
        self._min_duration = min_round_duration_sec

        # -- volatile state (reset per video) ---------------------------------
        self._round_counter = 0
        self._phase: str = "idle"            # idle | candidate_start | in_round | candidate_end
        self._debounce_count = 0
        self._transition_frame: int | None = None
        self._transition_ts: float | None = None
        self._last_start_ts: float | None = None
        self._pending_events: list[GameEvent] = []

    # -- EventEngine interface -----------------------------------------------

    def update(
        self, hud_state: HudState, tracks: Sequence[Track] = ()
    ) -> Sequence[GameEvent]:
        """Ingest one frame and return any newly confirmed events."""
        active = hud_state.values.get(self._state_key)
        is_active = bool(active) if active is not None else False
        fi = hud_state.frame_index
        ts = hud_state.timestamp_sec

        self._pending_events.clear()
        self._transition(is_active, fi, ts)
        return tuple(self._pending_events)

    def finalize(self) -> Sequence[GameEvent]:
        """Emit a ROUND_END if the video ended mid-round."""
        events: list[GameEvent] = []

        if self._phase in ("candidate_start", "in_round"):
            # We never saw the round end — force one.
            self._round_counter += 1
            rid = f"round_{self._round_counter:03d}"
            events.append(self._make_event(
                EventType.ROUND_END,
                rid,
                self._transition_ts or 0.0,
                fi=None,
            ))

        self._reset()
        return events

    # -- internal ------------------------------------------------------------

    def _transition(self, active: bool, fi: int, ts: float) -> None:
        """State machine: detect confirmed rising/falling edges of *active*."""

        if self._phase == "idle":
            if active:
                self._phase = "candidate_start"
                self._debounce_count = 1
                self._transition_frame = fi
                self._transition_ts = ts
                if self._debounce_count >= self._debounce:
                    self._confirm_round_start(fi, ts)

        elif self._phase == "candidate_start":
            if active:
                self._debounce_count += 1
                if self._debounce_count >= self._debounce:
                    self._confirm_round_start(fi, ts)
            else:
                # False alarm — reset.
                self._phase = "idle"
                self._debounce_count = 0

        elif self._phase == "in_round":
            if not active:
                self._phase = "candidate_end"
                self._debounce_count = 1
                self._transition_frame = fi
                self._transition_ts = ts
                if self._debounce_count >= self._debounce:
                    self._confirm_round_end(fi, ts)

        elif self._phase == "candidate_end":
            if not active:
                self._debounce_count += 1
                if self._debounce_count >= self._debounce:
                    self._confirm_round_end(fi, ts)
            else:
                # False alarm — back to in_round.
                self._phase = "in_round"
                self._debounce_count = 0

    def _confirm_round_start(self, fi: int, ts: float) -> None:
        self._round_counter += 1
        rid = f"round_{self._round_counter:03d}"
        start_ts = self._transition_ts if self._transition_ts is not None else ts
        self._last_start_ts = start_ts
        self._phase = "in_round"
        self._debounce_count = 0

        self._pending_events.append(
            self._make_event(EventType.ROUND_START, rid, start_ts, fi=fi)
        )

    def _confirm_round_end(self, fi: int, ts: float) -> None:
        end_ts = self._transition_ts if self._transition_ts is not None else ts
        rid = f"round_{self._round_counter:03d}"

        # Suppress implausibly short rounds.
        if (
            self._last_start_ts is not None
            and (end_ts - self._last_start_ts) < self._min_duration
        ):
            self._phase = "in_round"  # keep the round open
            self._debounce_count = 0
            return

        self._phase = "idle"
        self._debounce_count = 0

        self._pending_events.append(
            self._make_event(EventType.ROUND_END, rid, end_ts, fi=fi)
        )

    def _make_event(
        self,
        event_type: EventType,
        round_id: str,
        ts: float,
        fi: int | None,
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
                    source=f"RoundBoundaryDetector.{self._state_key}",
                )
            ],
            attributes={"round_id": round_id},
        )

    def _reset(self) -> None:
        self._round_counter = 0
        self._phase = "idle"
        self._debounce_count = 0
        self._transition_frame = None
        self._transition_ts = None
        self._last_start_ts = None
        self._pending_events.clear()
