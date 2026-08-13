from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gamesight.domain.models import Detection, EventType, RoundAnalysis
from gamesight.perception.cs2_detector import (
    CS2FactionDetector, PlayerDetectionSample, build_engagement_events,
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
    assert engagement.start_sec == 90.0
    assert engagement.evidence[0].frame_index == 2700
    assert first_visible.start_sec == 90.0
    assert engagement.attributes["enemy_team"] == "ct"


def test_unknown_player_team_never_guesses_enemy_identity():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=30)]
    sample = PlayerDetectionSample(30, 1.0, None, (
        Detection(label="ct", confidence=.9, bbox_xyxy=(1, 1, 5, 9),
                  frame_index=30, timestamp_sec=1),
    ))
    assert build_engagement_events(rounds, [sample]) == []
