"""Fast match-level regression checks using anonymized native HUD signals."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamesight.domain.models import EventType, GameEvent
from gamesight.events.aggregator import aggregate_events
from gamesight.perception.first_person import FirstPersonSample
from gamesight.perception.native_kill import detect_native_kills
from gamesight.perception.native_status import detect_native_deaths


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "test1_native_signals.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _boundary_event(
    event_type: EventType, timestamp: float, round_id: str,
) -> GameEvent:
    return GameEvent(
        event_id=f"fixture_{event_type.value}_{round_id}",
        event_type=event_type,
        start_sec=timestamp,
        confidence=1.0,
        attributes={"round_id": round_id},
    )


def _round_boundaries(data: dict) -> tuple[list, list[GameEvent]]:
    events: list[GameEvent] = []
    final_round = data["expected"]["open_final_round"]
    for index, spec in enumerate(data["rounds"]):
        start = index * 70.0
        events.append(_boundary_event(
            EventType.ROUND_START, start, spec["round_id"],
        ))
        if spec["round_id"] != final_round:
            events.append(_boundary_event(
                EventType.ROUND_END, start + 60.0, spec["round_id"],
            ))
    return aggregate_events(events), events


def _kill_sample(
    timestamp: float, *, score: float, fingerprint: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        frame_index=round(timestamp * 30),
        timestamp_sec=timestamp,
        local_kill_highlight=True,
        local_kill_highlight_score=score,
        local_kill_row_fingerprints=(
            (fingerprint.encode("utf-8"),) if fingerprint else ()
        ),
    )


def _kill_samples(data: dict, profile_name: str) -> list[SimpleNamespace]:
    profile = data["sampling_profiles"][profile_name]
    sparse = set(profile["sparse_strong_rounds"])
    step = float(profile["sample_step_sec"])
    samples: list[SimpleNamespace] = []
    starts = {
        spec["round_id"]: index * 70.0
        for index, spec in enumerate(data["rounds"])
    }
    for spec in data["rounds"]:
        round_id = spec["round_id"]
        for kill_index in range(spec["kills"]):
            timestamp = starts[round_id] + 10.0 + kill_index * 8.0
            fingerprint = f"{round_id}-kill-{kill_index}"
            if round_id in sparse and kill_index == 0:
                samples.append(_kill_sample(
                    timestamp, score=.99, fingerprint=fingerprint,
                ))
            else:
                samples.extend((
                    _kill_sample(
                        timestamp, score=.90, fingerprint=fingerprint,
                    ),
                    _kill_sample(
                        timestamp + step, score=.86,
                        fingerprint=fingerprint,
                    ),
                ))
    for artifact_index, artifact in enumerate(data["outline_artifacts"]):
        samples.append(_kill_sample(
            starts[artifact["round_id"]] + artifact["offset_sec"],
            score=artifact["score"],
            fingerprint=f"artifact-{artifact_index}",
        ))
    return sorted(samples, key=lambda sample: sample.timestamp_sec)


def _status_samples(data: dict) -> list[FirstPersonSample]:
    samples: list[FirstPersonSample] = []
    for index, spec in enumerate(data["rounds"]):
        start = index * 70.0
        for sample_index in range(120):
            offset = sample_index * .5
            death_gap = spec["died"] and 50.0 <= offset <= 50.5
            samples.append(FirstPersonSample(
                frame_index=round((start + offset) * 30),
                timestamp_sec=start + offset,
                flashed=False,
                scoped=False,
                motion_score=.1,
                health_hud_visible=not death_gap,
                health_hud_score=0.0 if death_gap else .8,
            ))
    return samples


def test_fixture_reconstructs_all_rounds_and_open_final_round():
    data = _fixture()
    rounds, _ = _round_boundaries(data)

    assert len(rounds) == data["expected"]["rounds"]
    assert rounds[-1].round_id == data["expected"]["open_final_round"]
    assert rounds[-1].end_sec is None


@pytest.mark.parametrize("profile_name", ["fps_10", "fps_2"])
def test_fixture_preserves_kills_across_sampling_profiles(profile_name: str):
    data = _fixture()
    rounds, _ = _round_boundaries(data)
    result = detect_native_kills(
        rounds, _kill_samples(data, profile_name), [],
    )

    assert len(result.events) == data["expected"]["kills"]
    counts = Counter(event.attributes["round_id"] for event in result.events)
    for round_id in data["expected"]["required_kill_rounds"]:
        assert counts[round_id] >= 1
    assert result.highlight_episodes > len(result.events)


def test_fixture_reconstructs_match_level_20_kills_and_10_deaths():
    data = _fixture()
    rounds, boundary_events = _round_boundaries(data)
    kills = detect_native_kills(
        rounds, _kill_samples(data, "fps_2"), [],
    )
    deaths = detect_native_deaths(rounds, _status_samples(data))
    reconstructed = aggregate_events([
        *boundary_events, *kills.events, *deaths.events,
    ])

    assert deaths.available
    assert len(reconstructed) == data["expected"]["rounds"]
    assert sum(
        event.event_type == EventType.PLAYER_KILL
        for round_ in reconstructed for event in round_.events
    ) == data["expected"]["kills"]
    assert sum(
        event.event_type == EventType.PLAYER_DEATH
        for round_ in reconstructed for event in round_.events
    ) == data["expected"]["deaths"]
