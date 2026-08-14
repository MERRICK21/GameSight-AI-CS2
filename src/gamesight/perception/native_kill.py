"""Conservative POV-kill attribution from native CS2 visual evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from gamesight.domain.models import EventType, Evidence, GameEvent, RoundAnalysis


@dataclass(frozen=True)
class NativeKillResult:
    """Kill events plus whether a local-kill signal was verifiable."""

    events: tuple[GameEvent, ...]
    available: bool
    highlight_episodes: int
    matched_episodes: int


def detect_native_kills(
    rounds: Sequence[RoundAnalysis],
    samples: Sequence[object],
    engagement_events: Sequence[GameEvent],
    *,
    merge_gap_sec: float = 2.1,
    post_engagement_sec: float = 3.0,
    row_retention_sec: float = 5.5,
    minimum_observation_samples: int = 2,
    single_sample_score_threshold: float = .97,
) -> NativeKillResult:
    """Count native local-kill rows and optionally corroborate engagements.

    CS2's red local-player outline is the ownership signal and therefore the
    source of truth for counting.  YOLO opposing-faction and POV-fire evidence
    raises confidence and improves review context, but sparse 2 FPS sampling
    must not veto a native HUD kill that was independently observed.
    """
    highlighted = sorted(
        (
            sample for sample in samples
            if bool(getattr(sample, "local_kill_highlight", False))
        ),
        key=lambda sample: float(getattr(sample, "timestamp_sec", 0.0)),
    )
    episodes: list[list[object]] = []
    for sample in highlighted:
        timestamp = float(getattr(sample, "timestamp_sec", 0.0))
        if (
            not episodes
            or timestamp
            - float(getattr(episodes[-1][-1], "timestamp_sec", 0.0))
            > merge_gap_sec
        ):
            episodes.append([sample])
        else:
            episodes[-1].append(sample)

    firefights = [
        event for event in engagement_events
        if event.event_type == EventType.ENGAGEMENT_CANDIDATE
        and event.attributes.get("engagement_level") == "likely_firefight"
        and event.attributes.get("first_shot_candidate_sec") is not None
    ]
    if minimum_observation_samples < 1:
        raise ValueError("minimum_observation_samples must be >= 1")
    if not 0.0 <= single_sample_score_threshold <= 1.0:
        raise ValueError("single_sample_score_threshold must be between 0 and 1")

    observations = _deduplicate_highlight_rows(
        episodes, row_retention_sec=row_retention_sec,
    )
    events: list[GameEvent] = []
    for episode in observations:
        first = episode[0]
        peak = max(episode, key=lambda sample: float(
            getattr(sample, "local_kill_highlight_score", 0.0)
        ))
        timestamp = float(getattr(first, "timestamp_sec", 0.0))
        round_analysis = _round_at(rounds, timestamp)
        if round_analysis is None:
            continue
        candidates: list[tuple[float, GameEvent]] = []
        for engagement in firefights:
            if engagement.attributes.get("round_id") != round_analysis.round_id:
                continue
            shot_sec = float(engagement.attributes["first_shot_candidate_sec"])
            engagement_end = float(engagement.end_sec or engagement.start_sec)
            if not (
                shot_sec - .75
                <= timestamp
                <= max(engagement_end + post_engagement_sec, shot_sec + 3.5)
            ):
                continue
            candidates.append((abs(timestamp - shot_sec), engagement))
        engagement = (
            min(candidates, key=lambda item: item[0])[1]
            if candidates else None
        )
        observation_frames = {
            int(getattr(sample, "frame_index", -1)) for sample in episode
        }
        score = float(getattr(peak, "local_kill_highlight_score", 0.0))
        # A native kill-feed row remains visible for multiple sampled frames.
        # A one-frame outline without separate engagement evidence is a HUD
        # transition/fading-contour artefact unless its complete native-frame
        # geometry is exceptionally strong.  The high-score escape hatch is
        # required at sparse 1--2 FPS sampling, where a real row may only land
        # on one decoded frame.  It is independent of optional YOLO.
        if (
            len(observation_frames) < minimum_observation_samples
            and engagement is None
            and score < single_sample_score_threshold
        ):
            continue
        confidence = min(.94, .80 + score * .12)
        if engagement is not None:
            confidence = min(.96, confidence + .04)
        event_index = len(events) + 1
        evidence = [Evidence(
            frame_index=int(getattr(first, "frame_index", 0)),
            timestamp_sec=timestamp,
            source="NativeKillDetector.local_kill_highlight",
        )]
        if engagement is not None:
            evidence.extend(engagement.evidence[:1])
        events.append(GameEvent(
            event_id=f"native_kill_{round_analysis.round_id}_{event_index:02d}",
            event_type=EventType.PLAYER_KILL,
            start_sec=timestamp,
            confidence=confidence,
            evidence=evidence,
            attributes={
                "round_id": round_analysis.round_id,
                "method": "native_local_kill_highlight",
                "classification": "native_personal_kill",
                "native_highlight_score": round(score, 4),
                "highlight_duration_sec": round(
                    float(getattr(episode[-1], "timestamp_sec", timestamp))
                    - timestamp,
                    3,
                ),
                "observation_sample_count": len(observation_frames),
                "single_sample_strong_geometry": (
                    len(observation_frames) < minimum_observation_samples
                    and score >= single_sample_score_threshold
                ),
                "engagement_corroborated": engagement is not None,
                "engagement_event_id": (
                    engagement.event_id if engagement is not None else None
                ),
                "first_shot_candidate_sec": (
                    engagement.attributes.get("first_shot_candidate_sec")
                    if engagement is not None else None
                ),
                "clip_trigger_sec": round(timestamp, 3),
            },
        ))

    return NativeKillResult(
        events=tuple(events),
        available=bool(events),
        highlight_episodes=len(episodes),
        matched_episodes=sum(
            bool(event.attributes.get("engagement_corroborated"))
            for event in events
        ),
    )


def _deduplicate_highlight_rows(
    episodes: Sequence[Sequence[object]], *, row_retention_sec: float,
) -> list[list[object]]:
    """Track unique highlighted feed rows across fades and position changes."""
    known: list[dict[str, object]] = []
    unknown: list[list[object]] = []
    for episode in episodes:
        episode_tracks: list[dict[str, object]] = []
        episode_row_limit = max(
            1,
            max(
                len(tuple(getattr(sample, "local_kill_row_fingerprints", ())))
                for sample in episode
            ),
        )
        for sample in episode:
            sample_tracks: list[dict[str, object]] = []
            fingerprints = tuple(getattr(
                sample, "local_kill_row_fingerprints", (),
            ))
            for fingerprint in fingerprints:
                timestamp = float(getattr(sample, "timestamp_sec", 0.0))
                matches = [
                    track for track in known
                    if timestamp - float(track["last_seen"]) <= row_retention_sec
                    and track not in sample_tracks
                    and any(
                        _row_fingerprints_match(fingerprint, previous)
                        for previous in track["fingerprints"]
                    )
                ]
                if matches:
                    track = max(matches, key=lambda item: float(item["last_seen"]))
                    track["last_seen"] = timestamp
                    track["samples"].append(sample)
                    if len(track["fingerprints"]) < 4:
                        track["fingerprints"].append(fingerprint)
                elif len(episode_tracks) < episode_row_limit:
                    track = {
                        "samples": [sample],
                        "fingerprints": [fingerprint],
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                    }
                    known.append(track)
                else:
                    # A partially faded outline can distort its crop enough to
                    # miss visual matching.  It cannot introduce a third kill
                    # when no frame in this continuous feed episode ever shows
                    # a third highlighted row.
                    track = max(
                        episode_tracks,
                        key=lambda item: float(item["last_seen"]),
                    )
                    track["last_seen"] = timestamp
                    track["samples"].append(sample)
                if track not in episode_tracks:
                    episode_tracks.append(track)
                sample_tracks.append(track)
        if not episode_tracks:
            unknown.append(list(episode))

    # A fading row occasionally has no extractable rectangle at all.  If that
    # orphan is close to a content-tracked row, treat it as the same feed item
    # instead of manufacturing another kill from the temporary detector gap.
    for episode in unknown:
        start = float(getattr(episode[0], "timestamp_sec", 0.0))
        end = float(getattr(episode[-1], "timestamp_sec", start))
        nearby = [
            track for track in known
            if start - row_retention_sec <= float(track["last_seen"])
            and float(track["first_seen"]) <= end + row_retention_sec
        ]
        if nearby:
            track = min(
                nearby,
                key=lambda item: min(
                    abs(start - float(item["last_seen"])),
                    abs(end - float(item["first_seen"])),
                ),
            )
            track["samples"].extend(episode)
            track["first_seen"] = min(float(track["first_seen"]), start)
            track["last_seen"] = max(float(track["last_seen"]), end)
        else:
            known.append({
                "samples": list(episode),
                "fingerprints": [],
                "first_seen": start,
                "last_seen": end,
            })

    result = [
        sorted(track["samples"], key=lambda sample: float(
            getattr(sample, "timestamp_sec", 0.0)
        ))
        for track in known
    ]
    result.sort(key=lambda episode: float(
        getattr(episode[0], "timestamp_sec", 0.0)
    ))
    return result


def _row_fingerprints_match(left: bytes, right: bytes) -> bool:
    """Match two encoded victim-side row crops without interpreting text."""
    if left == right:
        return True
    left_image = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_GRAYSCALE)
    if left_image is None or right_image is None:
        return False
    detector = cv2.SIFT_create()
    left_keypoints, left_descriptors = detector.detectAndCompute(left_image, None)
    right_keypoints, right_descriptors = detector.detectAndCompute(right_image, None)
    if left_descriptors is None or right_descriptors is None:
        return False
    matcher = cv2.BFMatcher()

    def score(query: np.ndarray, train: np.ndarray) -> tuple[int, float]:
        pairs = matcher.knnMatch(query, train, k=2)
        good = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance:
                good.append(pair[0])
        denominator = max(1, min(len(left_keypoints), len(right_keypoints)))
        return len(good), len(good) / denominator

    forward = score(left_descriptors, right_descriptors)
    reverse = score(right_descriptors, left_descriptors)
    return any(count >= 6 and ratio >= .20 for count, ratio in (forward, reverse))


def _round_at(
    rounds: Sequence[RoundAnalysis], timestamp: float,
) -> RoundAnalysis | None:
    for round_analysis in rounds:
        if (
            round_analysis.start_sec <= timestamp
            and (
                round_analysis.end_sec is None
                or timestamp < round_analysis.end_sec
            )
        ):
            return round_analysis
    return None
