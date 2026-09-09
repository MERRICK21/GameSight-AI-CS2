"""Optional CS2-specific faction detection and engagement-window events."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gamesight.domain.models import (
    Detection, EventType, Evidence, GameEvent, RoundAnalysis,
)


@dataclass(frozen=True)
class PlayerDetectionSample:
    frame_index: int
    timestamp_sec: float
    player_team: str | None
    detections: tuple[Detection, ...]


def inference_stride(sample_fps: float, target_fps: float = 2.0) -> int:
    """Return the sampled-frame stride needed to cap neural inference rate."""
    if sample_fps <= 0 or target_fps <= 0:
        raise ValueError("sample_fps and target_fps must be positive")
    return max(1, round(sample_fps / target_fps))


class CS2FactionDetector:
    """Detect T/CT bodies with a CS2-trained Ultralytics model.

    The model is injected in tests and loaded lazily in production.  Only
    body classes are retained; head classes would duplicate the same player.
    """

    _LABEL_TO_TEAM = {"c": "ct", "ct": "ct", "t": "t"}

    def __init__(
        self,
        model: object | None = None,
        model_path: str | Path | None = None,
        confidence_threshold: float = .30,
    ) -> None:
        self._confidence = confidence_threshold
        self._model = model if model is not None else self._load_model(model_path)

    def detect(
        self, image: np.ndarray, frame_index: int, timestamp_sec: float,
    ) -> tuple[Detection, ...]:
        if not isinstance(image, np.ndarray):
            return ()
        results = self._model(image, verbose=False, conf=self._confidence)
        result = results[0] if isinstance(results, (list, tuple)) else results
        if getattr(result, "boxes", None) is None:
            return ()
        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(_scalar(box.cls))
            confidence = float(_scalar(box.conf))
            raw_label = str(names.get(cls_id, cls_id)).lower()
            team = self._LABEL_TO_TEAM.get(raw_label)
            if team is None or confidence < self._confidence:
                continue
            xyxy = box.xyxy[0].tolist()
            detections.append(Detection(
                label=team,
                confidence=confidence,
                bbox_xyxy=tuple(float(value) for value in xyxy),
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            ))
        return tuple(detections)

    @staticmethod
    def _load_model(model_path: str | Path | None) -> object:
        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError(
                "CS2 detector weights are missing: models/yolov10n_cs2.pt"
            )
        config_dir = Path(model_path).resolve().parent / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "CS2 detection requires ultralytics and torch."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return YOLO(str(model_path), task="detect").to(device)


def build_engagement_events(
    rounds: list[RoundAnalysis],
    samples: Sequence[PlayerDetectionSample],
    visual_samples: Sequence[object] = (),
    max_episodes_per_round: int = 4,
    merge_gap_sec: float = 2.1,
) -> list[GameEvent]:
    """Turn confirmed opposing-faction sightings into reviewable episodes."""
    events: list[GameEvent] = []
    for round_analysis in rounds:
        end = round_analysis.end_sec
        if end is None:
            continue
        enemy_samples: list[tuple[PlayerDetectionSample, Detection]] = []
        for sample in samples:
            if not (round_analysis.start_sec <= sample.timestamp_sec < end):
                continue
            if sample.player_team not in {"t", "ct"}:
                continue
            enemies = [
                detection for detection in sample.detections
                if detection.label in {"t", "ct"}
                and detection.label != sample.player_team
            ]
            if enemies:
                enemy_samples.append((sample, max(enemies, key=lambda item: item.confidence)))
        if not enemy_samples:
            continue

        episodes: list[list[tuple[PlayerDetectionSample, Detection]]] = []
        active: list[tuple[PlayerDetectionSample, Detection]] = []
        for item in enemy_samples:
            if active and item[0].timestamp_sec - active[-1][0].timestamp_sec > merge_gap_sec:
                episodes.append(active)
                active = []
            active.append(item)
        if active:
            episodes.append(active)

        for episode_index, episode in enumerate(
            episodes[:max_episodes_per_round], start=1,
        ):
            best_sample, best_detection = max(
                episode, key=lambda item: item[1].confidence,
            )
            first_sample = episode[0][0]
            last_sample = episode[-1][0]
            nearby_signals = [
                sample for sample in visual_samples
                if first_sample.timestamp_sec - 1.5
                <= float(getattr(sample, "timestamp_sec", -1.0))
                <= last_sample.timestamp_sec + 1.5
            ]
            shot_signals = [
                sample for sample in nearby_signals
                if bool(getattr(sample, "shot_candidate", False))
            ]
            damage_signals = [
                sample for sample in nearby_signals
                if bool(getattr(sample, "damage_candidate", False))
            ]
            combat_signals = shot_signals + damage_signals
            signal_sample = max(
                combat_signals,
                key=lambda sample: max(
                    float(getattr(sample, "shot_signal_score", 0.0)),
                    float(getattr(sample, "damage_signal_score", 0.0)),
                ),
                default=None,
            )
            likely_firefight = signal_sample is not None
            first_shot_sec = (
                min(float(getattr(sample, "timestamp_sec")) for sample in shot_signals)
                if shot_signals else None
            )
            first_damage_sec = (
                min(float(getattr(sample, "timestamp_sec")) for sample in damage_signals)
                if damage_signals else None
            )
            clip_trigger_sec = (
                float(getattr(signal_sample, "timestamp_sec"))
                if signal_sample is not None else best_sample.timestamp_sec
            )
            attributes = {
                "round_id": round_analysis.round_id,
                "player_team": first_sample.player_team,
                "enemy_team": best_detection.label,
                "max_confidence": round(best_detection.confidence, 4),
                "bbox_xyxy": ",".join(
                    str(round(value, 1)) for value in best_detection.bbox_xyxy
                ),
                "engagement_level": (
                    "likely_firefight" if likely_firefight else "visual_contact"
                ),
                "shot_candidate_count": len(shot_signals),
                "damage_candidate_count": len(damage_signals),
                "clip_trigger_sec": round(clip_trigger_sec, 3),
                "first_visible_sec": round(first_sample.timestamp_sec, 3),
                "last_visible_sec": round(last_sample.timestamp_sec, 3),
                "visible_sample_count": len(episode),
                "observed_span_sec": round(
                    max(0.0, last_sample.timestamp_sec - first_sample.timestamp_sec),
                    3,
                ),
                "first_shot_candidate_sec": (
                    round(first_shot_sec, 3) if first_shot_sec is not None else None
                ),
                "first_damage_candidate_sec": (
                    round(first_damage_sec, 3)
                    if first_damage_sec is not None else None
                ),
                "first_shot_offset_sec": (
                    round(first_shot_sec - first_sample.timestamp_sec, 3)
                    if first_shot_sec is not None else None
                ),
            }
            event_id = (
                f"engagement_{round_analysis.round_id}_{episode_index:02d}"
            )
            evidence = [Evidence(
                frame_index=best_sample.frame_index,
                timestamp_sec=best_sample.timestamp_sec,
                source="CS2FactionDetector.opposing_faction",
            )]
            if signal_sample is not None:
                evidence.append(Evidence(
                    frame_index=int(getattr(signal_sample, "frame_index")),
                    timestamp_sec=float(getattr(signal_sample, "timestamp_sec")),
                    source="FirstPersonAnalyzer.combat_visual_candidate",
                ))
            events.append(GameEvent(
                event_id=event_id,
                event_type=EventType.ENGAGEMENT_CANDIDATE,
                start_sec=first_sample.timestamp_sec,
                end_sec=last_sample.timestamp_sec,
                confidence=min(.92, .58 + best_detection.confidence * .4),
                evidence=evidence,
                attributes=attributes,
            ))
            if episode_index == 1:
                events.append(GameEvent(
                    event_id=f"enemy_first_visible_{round_analysis.round_id}",
                    event_type=EventType.ENEMY_FIRST_VISIBLE,
                    start_sec=first_sample.timestamp_sec,
                    confidence=min(.92, .58 + best_detection.confidence * .4),
                    evidence=evidence,
                    attributes=attributes,
                ))
    return events


def _scalar(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "__getitem__"):
        return float(value[0])  # type: ignore[index]
    return float(value)  # type: ignore[arg-type]
