"""Tests for conservative native health-HUD death transitions."""

from gamesight.domain.models import EventType, RoundAnalysis
from gamesight.perception.first_person import FirstPersonSample
from gamesight.perception.native_status import detect_native_deaths


def _sample(
    timestamp: float, *, visible: bool, flashed: bool = False,
    damage: bool = False, left_card: bool | None = None,
    left_score: float = 0.0, right_card: bool | None = None,
    right_score: float = 0.0,
) -> FirstPersonSample:
    return FirstPersonSample(
        frame_index=int(timestamp * 30),
        timestamp_sec=timestamp,
        flashed=flashed,
        scoped=False,
        motion_score=0.1,
        damage_candidate=damage,
        health_hud_visible=visible,
        health_hud_score=.08 if visible else 0.0,
        player_card_left_alive=left_card,
        player_card_right_alive=right_card,
        player_card_left_selected_score=left_score,
        player_card_right_selected_score=right_score,
    )


def test_stable_hud_then_sustained_loss_emits_one_death():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)]
    samples = [
        _sample(t / 2, visible=not (10 <= t <= 11), damage=(t == 10))
        for t in range(20)
    ]
    result = detect_native_deaths(rounds, samples)
    assert result.available
    assert len(result.events) == 1
    event = result.events[0]
    assert event.event_type == EventType.PLAYER_DEATH
    assert event.start_sec == 5.0
    assert event.attributes["method"] == "native_health_hud_disappearance"
    assert event.attributes["hud_missing_duration_sec"] == 1.0
    assert event.attributes["damage_candidate_nearby"] is True
    assert event.confidence == .9


def test_flash_frames_do_not_become_hud_loss():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)]
    samples = [
        _sample(
            t / 2,
            visible=not (10 <= t <= 12),
            flashed=(10 <= t <= 12),
        )
        for t in range(20)
    ]
    result = detect_native_deaths(rounds, samples)
    assert result.available
    assert result.events == ()


def test_single_missing_sample_is_rejected():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)]
    samples = [
        _sample(t / 2, visible=(t != 10)) for t in range(20)
    ]
    result = detect_native_deaths(rounds, samples)
    assert result.available
    assert result.events == ()


def test_single_missing_sample_with_damage_candidate_is_retained():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)]
    samples = [
        _sample(t / 2, visible=(t != 10), damage=(t == 9))
        for t in range(20)
    ]
    result = detect_native_deaths(rounds, samples)
    assert result.available
    assert len(result.events) == 1
    assert result.events[0].confidence == .9


def test_flash_can_bridge_prior_visible_hud_to_death_transition():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=12)]
    samples = []
    for t in range(24):
        timestamp = t / 2
        flashed = 12 <= t <= 15
        missing_after_flash = 16 <= t <= 18
        samples.append(_sample(
            timestamp,
            visible=not flashed and not missing_after_flash,
            flashed=flashed,
        ))
    result = detect_native_deaths(rounds, samples)
    assert result.available
    assert len(result.events) == 1
    assert result.events[0].attributes["flash_bridge"] is True


def test_long_unsupported_hud_loss_is_rejected():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=14)]
    samples = [
        _sample(t / 2, visible=not (10 <= t <= 22)) for t in range(28)
    ]
    result = detect_native_deaths(rounds, samples)
    assert result.available
    assert result.events == ()


def test_unsupported_layout_keeps_deaths_unavailable():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)]
    samples = [_sample(t / 2, visible=False) for t in range(20)]
    result = detect_native_deaths(rounds, samples)
    assert not result.available
    assert result.events == ()


def test_selected_native_player_card_recovers_death_after_spectator_switch():
    rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)]
    samples = [
        _sample(
            t / 2,
            # Bottom health remains visible because the recording immediately
            # switches to another first-person player after the POV death.
            visible=True,
            left_card=not (10 <= t <= 13),
            left_score=.55,
            right_card=True,
            right_score=.04,
        )
        for t in range(20)
    ]

    result = detect_native_deaths(rounds, samples)

    assert len(result.events) == 1
    assert result.events[0].start_sec == 5.0
    assert result.events[0].attributes["method"] == (
        "native_player_card_disappearance"
    )
    assert result.events[0].attributes["selected_card_side"] == "left"
