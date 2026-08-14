from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gamesight.domain.models import Detection, EventType, RoundAnalysis
from gamesight.perception.cs2_detector import (
    CS2FactionDetector, PlayerDetectionSample, build_engagement_events,
    inference_stride,
)


class _Model:
    names = {0: "c", 1: "ch", 2: "t", 3: "th"}

    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, image, **kwargs):
        return [SimpleNamespace(boxes=self._boxes, names=self.names)]


def _box(cls_id, confidence, xyxy):
    return SimpleNamespace(
        cls=np.array([cls_id]), conf=np.array([confidence]),
        xyxy=np.array([xyxy], dtype=float),
    )


def test_detector_maps_body_classes_and_ignores_duplicate_heads():
    detector = CS2FactionDetector(model=_Model([
        _box(0, .8, [10, 20, 30, 80]),
        _box(1, .7, [15, 20, 22, 30]),
        _box(2, .9, [100, 20, 140, 90]),
    ]))
    result = detector.detect(np.zeros((100, 160, 3), dtype=np.uint8), 30, 1.0)
    assert [item.label for item in result] == ["ct", "t"]


def test_builds_enemy_engagement_but_not_teammate_episode():
    rounds = [RoundAnalysis(round_id="round_001", start_sec=50, end_sec=100)]
    samples = [
        PlayerDetectionSample(1800, 60.0, "t", (
            Detection(label="t", confidence=.95, bbox_xyxy=(1, 1, 5, 9),
                      frame_index=1800, timestamp_sec=60),
        )),
        PlayerDetectionSample(2670, 89.0, "t", (
            Detection(label="ct", confidence=.42, bbox_xyxy=(10, 10, 30, 80),
                      frame_index=2670, timestamp_sec=89),
        )),
        PlayerDetectionSample(2700, 90.0, "t", (
            Detection(label="ct", confidence=.83, bbox_xyxy=(12, 10, 32, 80),
                      frame_index=2700, timestamp_sec=90),
        )),
    ]
    events = build_engagement_events(rounds, samples)
    engagement = next(
        item for item in events
        if item.event_type == EventType.ENGAGEMENT_CANDIDATE
    )
    first_visible = next(
        item for item in events
        if item.event_type == EventType.ENEMY_FIRST_VISIBLE
    )
    assert engagement.start_sec == 89.0
    assert engagement.evidence[0].frame_index == 2700
    assert first_visible.start_sec == 89.0
    assert engagement.attributes["enemy_team"] == "ct"
    assert engagement.attributes["engagement_level"] == "visual_contact"
    assert engagement.attributes["visible_sample_count"] == 2
    assert engagement.attributes["observed_span_sec"] == 1.0


def test_visual_combat_signal_upgrades_contact_to_likely_firefight():
    rounds = [RoundAnalysis(round_id="round_002", start_sec=56, end_sec=100)]
    samples = [
        PlayerDetectionSample(2610, 87.0, "t", (
            Detection(label="ct", confidence=.8, bbox_xyxy=(10, 10, 30, 80),
                      frame_index=2610, timestamp_sec=87.0),
        )),
    ]
    visual_samples = [
        SimpleNamespace(
            frame_index=2625, timestamp_sec=87.5,
            shot_candidate=True, damage_candidate=False,
            shot_signal_score=.09, damage_signal_score=0.0,
        ),
    ]
    events = build_engagement_events(rounds, samples, visual_samples)
    engagement = next(
        item for item in events
        if item.event_type == EventType.ENGAGEMENT_CANDIDATE
    )
    assert engagement.attributes["engagement_level"] == "likely_firefight"
    assert engagement.attributes["shot_candidate_count"] == 1
    assert engagement.attributes["damage_candidate_count"] == 0
    assert engagement.attributes["clip_trigger_sec"] == 87.5
    assert engagement.attributes["first_visible_sec"] == 87.0
    assert engagement.attributes["last_visible_sec"] == 87.0
    assert engagement.attributes["visible_sample_count"] == 1
    assert engagement.attributes["observed_span_sec"] == 0.0
    assert engagement.attributes["first_shot_candidate_sec"] == 87.5
    assert engagement.attributes["first_shot_offset_sec"] == 0.5
    assert len(engagement.evidence) == 2


def test_unknown_player_team_never_guesses_enemy_identity():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=30)]
    sample = PlayerDetectionSample(30, 1.0, None, (
        Detection(label="ct", confidence=.9, bbox_xyxy=(1, 1, 5, 9),
                  frame_index=30, timestamp_sec=1),
    ))
    assert build_engagement_events(rounds, [sample]) == []


def test_neural_inference_is_capped_when_visual_sampling_is_high():
    assert inference_stride(2.0, 2.0) == 1
    assert inference_stride(10.0, 2.0) == 5


def test_inference_stride_rejects_invalid_rates():
    try:
        inference_stride(0.0, 2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero sample FPS must be rejected")
