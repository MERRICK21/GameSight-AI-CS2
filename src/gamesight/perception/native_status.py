"""Conservative POV death events from the native CS2 status HUD."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from gamesight.domain.models import EventType, Evidence, GameEvent, RoundAnalysis
from gamesight.perception.first_person import FirstPersonSample


@dataclass(frozen=True)
class NativeDeathResult:
    events: tuple[GameEvent, ...]
    available: bool
    hud_coverage: float
    eligible_rounds: int


def detect_native_deaths(
    rounds: Sequence[RoundAnalysis],
    samples: Sequence[FirstPersonSample],
    *,
    min_missing_sec: float = .75,
    max_missing_sec: float = 5.0,
    min_prior_visible_sec: float = 1.0,
) -> NativeDeathResult:
    """Detect one probable POV death per round from an evidenced HUD loss.

    A candidate requires a stable visible native health cluster, followed by
    sustained disappearance. Full-screen flash frames are ignored. Long HUD
    absences are rejected because they are more likely menus, edits, or an
    unsupported layout than a short CS2 death transition.
    """
    ordered = sorted(samples, key=lambda sample: sample.timestamp_sec)
    completed = [round_ for round_ in rounds if round_.end_sec is not None]
    total_non_flash = sum(not sample.flashed for sample in ordered)
    visible_non_flash = sum(
        sample.health_hud_visible for sample in ordered if not sample.flashed
    )
    coverage = (
        visible_non_flash / total_non_flash if total_non_flash else 0.0
    )
    events: list[GameEvent] = []
    eligible_rounds = 0

    for round_analysis in completed:
        assert round_analysis.end_sec is not None
        current = [
            sample for sample in ordered
            if round_analysis.start_sec <= sample.timestamp_sec
            < round_analysis.end_sec and not sample.flashed
        ]
        if len(current) < 4:
            continue
        gaps = [
            later.timestamp_sec - earlier.timestamp_sec
            for earlier, later in zip(current, current[1:])
            if later.timestamp_sec > earlier.timestamp_sec
        ]
        step = float(median(gaps)) if gaps else .5
        if sum(sample.health_hud_visible for sample in current) < 2:
            continue
        eligible_rounds += 1

        round_event_added = False
        index = 0
        while index < len(current):
            if current[index].health_hud_visible:
                index += 1
                continue
            start = index
            while index < len(current) and not current[index].health_hud_visible:
                index += 1
            run = current[start:index]
            missing_sec = (run[-1].timestamp_sec - run[0].timestamp_sec) + step
            bridging_flash = any(
                sample.flashed
                and run[0].timestamp_sec - 4.0 <= sample.timestamp_sec
                < run[0].timestamp_sec
                for sample in ordered
            )
            prior_lookback = 4.0 if bridging_flash else 2.0
            prior_window_start = run[0].timestamp_sec - max(
                prior_lookback, min_prior_visible_sec + step,
            )
            prior = [
                sample for sample in current[:start]
                if sample.timestamp_sec >= prior_window_start
                and sample.health_hud_visible
            ]
            prior_visible_sec = len(prior) * step
            after_opening = (
                run[0].timestamp_sec - round_analysis.start_sec >= 3.0
            )
            nearby_damage = any(
                sample.damage_candidate
                and abs(sample.timestamp_sec - run[0].timestamp_sec) <= 2.0
                for sample in current
            )
            # A one-sample loss is normally rejected.  Retain it only when a
            # separate two-sided damage-overlay candidate occurs nearby; this
            # recovers very short death animations at a round boundary.
            duration_supported = missing_sec >= min_missing_sec or (
                missing_sec >= step and nearby_damage
            )
            if not (
                after_opening
                and duration_supported
                and missing_sec <= max_missing_sec
                and prior_visible_sec >= min_prior_visible_sec
            ):
                continue
            confidence = .9 if nearby_damage else .84
            event_index = len(events) + 1
            events.append(GameEvent(
                event_id=f"native_player_death_{event_index:03d}",
                event_type=EventType.PLAYER_DEATH,
                start_sec=run[0].timestamp_sec,
                confidence=confidence,
                evidence=[Evidence(
                    frame_index=run[0].frame_index,
                    timestamp_sec=run[0].timestamp_sec,
                    source="NativeStatusDetector.health_hud_disappearance",
                )],
                attributes={
                    "round_id": round_analysis.round_id,
                    "method": "native_health_hud_disappearance",
                    "hud_missing_duration_sec": round(missing_sec, 3),
                    "prior_visible_duration_sec": round(prior_visible_sec, 3),
                    "damage_candidate_nearby": nearby_damage,
                    "flash_bridge": bridging_flash,
                    "clip_trigger_sec": round(run[0].timestamp_sec, 3),
                },
            ))
            round_event_added = True
            break

        if round_event_added:
            continue

        # A clipped/demo recording can switch to another first-person HUD
        # immediately after the POV death, so the bottom HP cluster remains
        # visible.  The native top roster still dims the selected POV card.
        # Pick the card side from its median orange selection highlight within
        # this round; no fixed halftime timestamp or player-name OCR is used.
        left_score = median([
            float(sample.player_card_left_selected_score)
            for sample in current
            if sample.player_card_left_alive is not None
        ] or [0.0])
        right_score = median([
            float(sample.player_card_right_selected_score)
            for sample in current
            if sample.player_card_right_alive is not None
        ] or [0.0])
        card_attribute = (
            "player_card_left_alive"
            if left_score > right_score else "player_card_right_alive"
        )
        selected_score = max(left_score, right_score)
        if selected_score < .12:
            continue
        card_values = [getattr(sample, card_attribute) for sample in current]
        if sum(value is True for value in card_values) < 2:
            continue

        index = 0
        while index < len(current):
            if card_values[index] is not False:
                index += 1
                continue
            start = index
            while index < len(current) and card_values[index] is False:
                index += 1
            run = current[start:index]
            missing_sec = (run[-1].timestamp_sec - run[0].timestamp_sec) + step
            prior_window_start = run[0].timestamp_sec - max(
                2.0, min_prior_visible_sec + step,
            )
            prior = [
                sample for sample in current[:start]
                if sample.timestamp_sec >= prior_window_start
                and getattr(sample, card_attribute) is True
            ]
            prior_visible_sec = len(prior) * step
            if not (
                run[0].timestamp_sec - round_analysis.start_sec >= 3.0
                and missing_sec >= min_missing_sec
                and missing_sec <= max_missing_sec
                and prior_visible_sec >= min_prior_visible_sec
            ):
                continue
            event_index = len(events) + 1
            events.append(GameEvent(
                event_id=f"native_player_death_{event_index:03d}",
                event_type=EventType.PLAYER_DEATH,
                start_sec=run[0].timestamp_sec,
                confidence=.90,
                evidence=[Evidence(
                    frame_index=run[0].frame_index,
                    timestamp_sec=run[0].timestamp_sec,
                    source="NativeStatusDetector.player_card_disappearance",
                )],
                attributes={
                    "round_id": round_analysis.round_id,
                    "method": "native_player_card_disappearance",
                    "hud_missing_duration_sec": round(missing_sec, 3),
                    "prior_visible_duration_sec": round(prior_visible_sec, 3),
                    "selected_card_side": (
                        "left" if card_attribute.endswith("left_alive")
                        else "right"
                    ),
                    "selected_card_score": round(selected_score, 4),
                    "clip_trigger_sec": round(run[0].timestamp_sec, 3),
                },
            ))
            break

    # Availability means the native signal was present across a meaningful
    # share of the recording and in most completed rounds.  Unsupported HUD
    # themes therefore keep death values unavailable instead of showing zero.
    eligible_ratio = eligible_rounds / len(completed) if completed else 0.0
    available = coverage >= .35 and eligible_ratio >= .6
    return NativeDeathResult(
        events=tuple(events if available else ()),
        available=available,
        hud_coverage=round(coverage, 4),
        eligible_rounds=eligible_rounds,
    )
