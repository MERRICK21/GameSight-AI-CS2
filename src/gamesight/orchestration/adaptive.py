"""Candidate-window selection for the two-stage video analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True, order=True)
class TimeWindow:
    start_sec: float
    end_sec: float


def merge_time_windows(
    windows: Iterable[TimeWindow],
    *,
    duration_sec: float | None = None,
    merge_gap_sec: float = 0.75,
) -> list[TimeWindow]:
    """Clamp, sort and merge overlapping/nearby windows."""
    clamped: list[TimeWindow] = []
    for window in windows:
        start = max(0.0, float(window.start_sec))
        end = max(start, float(window.end_sec))
        if duration_sec is not None:
            start = min(start, duration_sec)
            end = min(end, duration_sec)
        if end > start:
            clamped.append(TimeWindow(start, end))
    clamped.sort()
    merged: list[TimeWindow] = []
    for window in clamped:
        if not merged or window.start_sec > merged[-1].end_sec + merge_gap_sec:
            merged.append(window)
            continue
        previous = merged[-1]
        merged[-1] = TimeWindow(
            previous.start_sec, max(previous.end_sec, window.end_sec)
        )
    return merged


def build_refinement_windows(
    first_person_samples: Sequence[object],
    player_detection_samples: Sequence[object] = (),
    *,
    duration_sec: float | None = None,
    before_sec: float = 2.0,
    after_sec: float = 4.0,
) -> list[TimeWindow]:
    """Build high-rate windows from conservative, auditable visual signals."""
    timestamps: list[float] = []
    previous_health_visible: bool | None = None
    for sample in first_person_samples:
        health_visible = bool(getattr(sample, "health_hud_visible", False))
        health_disappeared = previous_health_visible is True and not health_visible
        previous_health_visible = health_visible
        if any((
            bool(getattr(sample, "flashed", False)),
            bool(getattr(sample, "scoped", False)),
            bool(getattr(sample, "shot_candidate", False)),
            bool(getattr(sample, "damage_candidate", False)),
            bool(getattr(sample, "local_kill_highlight", False)),
            health_disappeared,
        )):
            timestamps.append(float(getattr(sample, "timestamp_sec")))
    for sample in player_detection_samples:
        if getattr(sample, "detections", ()):
            timestamps.append(float(getattr(sample, "timestamp_sec")))
    return merge_time_windows(
        (
            TimeWindow(timestamp - before_sec, timestamp + after_sec)
            for timestamp in timestamps
        ),
        duration_sec=duration_sec,
    )


T = TypeVar("T")


def merge_samples(base: Sequence[T], refined: Sequence[T]) -> list[T]:
    """Merge frame-indexed samples, preferring higher-rate refined samples."""
    by_frame = {int(getattr(sample, "frame_index")): sample for sample in base}
    by_frame.update({
        int(getattr(sample, "frame_index")): sample for sample in refined
    })
    return sorted(by_frame.values(), key=lambda sample: int(
        getattr(sample, "frame_index")
    ))
