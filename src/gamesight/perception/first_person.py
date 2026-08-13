"""First-person visual signals that do not depend on player identity.

Only the gameplay viewport and native screen effects are measured.  Bottom
overlays, creator watermarks, names, chat, and kill-feed text are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import EventType, Evidence, GameEvent, RoundAnalysis


@dataclass(frozen=True)
class FirstPersonSample:
    frame_index: int
    timestamp_sec: float
    flashed: bool
    scoped: bool
    motion_score: float | None
    player_team: str | None = None


class FirstPersonAnalyzer:
    """Measure flash exposure, scope state, and camera motion per frame."""

    def __init__(self) -> None:
        self._previous: NDArray[np.uint8] | None = None
        self._previous_ts: float | None = None
        self._player_team: str | None = None

    def update(
        self, image: NDArray[np.uint8], frame_index: int, timestamp_sec: float
    ) -> FirstPersonSample:
        # Full-HD colour conversion on every 10 FPS sample is unnecessary for
        # screen-wide flash/scope geometry.  A 640px working frame preserves
        # the signal while cutting per-frame pixel work by roughly 9x at 1080p.
        source_h, source_w = image.shape[:2]
        if source_w > 640:
            working = cv2.resize(
                image,
                (640, max(1, int(round(source_h * 640 / source_w)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        mean = float(gray.mean())
        std = float(gray.std())
        flashed = mean > 205.0 and std < 65.0

        # Four off-centre patches sit outside the circular AWP viewport.  All
        # become black while scoped; requiring all four avoids dark-map noise.
        patches = (
            gray[int(h * .18):int(h * .38), int(w * .08):int(w * .27)],
            gray[int(h * .18):int(h * .38), int(w * .73):int(w * .92)],
            gray[int(h * .62):int(h * .82), int(w * .08):int(w * .27)],
            gray[int(h * .62):int(h * .82), int(w * .73):int(w * .92)],
        )
        dark_ratio = float(np.mean([np.mean(patch < 18) for patch in patches]))
        scoped = dark_ratio > 0.72 and mean < 105.0

        # Central gameplay viewport explicitly excludes all HUD edges and the
        # bottom watermark band.  Normalise by elapsed time for FPS stability.
        viewport = gray[int(h * .16):int(h * .84), int(w * .10):int(w * .90)]
        small = cv2.resize(viewport, (160, 90), interpolation=cv2.INTER_AREA)
        motion: float | None = None
        if self._previous is not None and self._previous_ts is not None:
            delta_sec = max(timestamp_sec - self._previous_ts, 0.05)
            difference = float(cv2.absdiff(small, self._previous).mean()) / 255.0
            motion = min(1.0, difference / delta_sec)
        self._previous = small
        self._previous_ts = timestamp_sec

        # The native CS2 team emblem sits at bottom-centre.  Restricting this
        # classifier to that HUD element avoids creator watermarks and names.
        detected_team = _detect_player_team(working)
        if detected_team is not None:
            self._player_team = detected_team

        return FirstPersonSample(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            flashed=flashed,
            scoped=scoped,
            motion_score=motion,
            player_team=self._player_team,
        )


def _detect_player_team(image: NDArray[np.uint8]) -> str | None:
    """Infer the POV side from the native bottom-centre team-colour HUD."""
    h, w = image.shape[:2]
    roi = image[int(h * .86):int(h * .995), int(w * .43):int(w * .57)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    terrorist = int(np.count_nonzero(cv2.inRange(
        hsv, np.array((10, 80, 80)), np.array((45, 255, 255)),
    )))
    counter_terrorist = int(np.count_nonzero(cv2.inRange(
        hsv, np.array((75, 70, 70)), np.array((130, 255, 255)),
    )))
    minimum = max(12, int(roi.shape[0] * roi.shape[1] * .001))
    if terrorist >= minimum and terrorist > counter_terrorist * 1.8:
        return "t"
    if counter_terrorist >= minimum and counter_terrorist > terrorist * 1.8:
        return "ct"
    return None


def build_first_person_summary_events(
    rounds: list[RoundAnalysis], samples: list[FirstPersonSample]
) -> list[GameEvent]:
    """Build a neutral round summary plus timestamped high-confidence moments.

    Raw scene motion is intentionally descriptive.  Running, turning, weapon
    switching, and actual aim corrections all move the viewport, so motion by
    itself must never be presented as an aim error or an engagement.
    """
    events: list[GameEvent] = []
    for round_analysis in rounds:
        if round_analysis.end_sec is None:
            continue
        current = [
            sample for sample in samples
            if round_analysis.start_sec <= sample.timestamp_sec < round_analysis.end_sec
        ]
        if not current:
            continue

        gaps = [
            later.timestamp_sec - earlier.timestamp_sec
            for earlier, later in zip(current, current[1:])
            if later.timestamp_sec > earlier.timestamp_sec
        ]
        step = float(np.median(gaps)) if gaps else 0.0
        flash_sec = round(sum(sample.flashed for sample in current) * step, 2)
        scoped_sec = round(sum(sample.scoped for sample in current) * step, 2)
        duration = max(round_analysis.end_sec - round_analysis.start_sec, 0.001)

        valid_motion = [
            sample for sample in current
            if sample.motion_score is not None and not sample.flashed
        ]
        motion_avg = (
            float(np.mean([sample.motion_score for sample in valid_motion]))
            if valid_motion else 0.0
        )
        stationary_ratio = (
            sum(sample.motion_score < 0.12 for sample in valid_motion)
            / len(valid_motion) if valid_motion else 0.0
        )

        flash_count = 0
        previously_flashed = False
        for sample in current:
            if sample.flashed and not previously_flashed:
                flash_count += 1
            previously_flashed = sample.flashed

        # A summary frame should represent the playable part of the round,
        # never simply the maximum-motion opening frame.  Use the first
        # high-confidence visual moment, otherwise a mid-round frame after
        # CS2's typical 15-second opening traversal window.
        moment_samples = [sample for sample in current if sample.flashed or sample.scoped]
        target_sec = round_analysis.start_sec + max(15.0, duration * 0.55)
        notable = moment_samples[0] if moment_samples else min(
            current, key=lambda sample: abs(sample.timestamp_sec - target_sec),
        )

        index = len(events) + 1
        events.append(GameEvent(
            event_id=f"first_person_summary_{index:03d}",
            event_type=EventType.FIRST_PERSON_SUMMARY,
            start_sec=notable.timestamp_sec,
            confidence=0.88,
            evidence=[Evidence(
                frame_index=notable.frame_index,
                timestamp_sec=notable.timestamp_sec,
                source="FirstPersonAnalyzer.gameplay_viewport",
            )],
            attributes={
                "round_id": round_analysis.round_id,
                "flash_count": flash_count,
                "flash_exposure_sec": flash_sec,
                "scoped_sec": scoped_sec,
                "scoped_ratio": round(scoped_sec / duration, 4),
                "view_motion_avg": round(motion_avg, 4),
                "stationary_ratio": round(stationary_ratio, 4),
                "motion_is_descriptive": True,
            },
        ))

        # Preserve individual flash/scope episodes so one round can yield
        # several reviewable moments instead of a single aggregate card.
        episodes = [
            ("flash", episode) for episode in _episodes(current, "flashed")
        ] + [
            ("scope", episode) for episode in _episodes(current, "scoped")
        ]
        episodes.sort(key=lambda item: item[1][0].timestamp_sec)
        for moment_index, (kind, episode) in enumerate(episodes[:4], start=1):
            episode_duration = max(
                step,
                episode[-1].timestamp_sec - episode[0].timestamp_sec + step,
            )
            # Ignore single noisy samples while retaining meaningful flashes;
            # scope advice needs a longer, continuous hold.
            minimum = 1.0 if kind == "flash" else 4.0
            if episode_duration < minimum:
                continue
            first = episode[0]
            events.append(GameEvent(
                event_id=(
                    f"first_person_{kind}_{round_analysis.round_id}_{moment_index:02d}"
                ),
                event_type=EventType.FIRST_PERSON_MOMENT,
                start_sec=first.timestamp_sec,
                end_sec=round(first.timestamp_sec + episode_duration, 2),
                confidence=0.88 if kind == "flash" else 0.86,
                evidence=[Evidence(
                    frame_index=first.frame_index,
                    timestamp_sec=first.timestamp_sec,
                    source=f"FirstPersonAnalyzer.{kind}_episode",
                )],
                attributes={
                    "round_id": round_analysis.round_id,
                    "moment_kind": kind,
                    "duration_sec": round(episode_duration, 2),
                },
            ))
    return events


def _episodes(
    samples: list[FirstPersonSample], attribute: str
) -> list[list[FirstPersonSample]]:
    """Group consecutive true samples into visual-effect episodes."""
    result: list[list[FirstPersonSample]] = []
    active: list[FirstPersonSample] = []
    for sample in samples:
        if bool(getattr(sample, attribute)):
            active.append(sample)
        elif active:
            result.append(active)
            active = []
    if active:
        result.append(active)
    return result
