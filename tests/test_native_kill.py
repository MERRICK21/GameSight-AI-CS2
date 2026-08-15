from types import SimpleNamespace

from gamesight.domain.models import EventType, Evidence, GameEvent, RoundAnalysis
from gamesight.perception.native_kill import detect_native_kills


def _sample(
    timestamp, *, highlighted=False, score=0.0, fingerprints=(), positions=(),
):
    return SimpleNamespace(
        frame_index=int(timestamp * 30),
        timestamp_sec=timestamp,
        local_kill_highlight=highlighted,
        local_kill_highlight_score=score,
        local_kill_row_fingerprints=fingerprints,
        local_kill_row_positions=positions,
    )


def _engagement(start=20.0, end=22.0, shot=21.5, level="likely_firefight"):
    return GameEvent(
        event_id="engagement_round_001_01",
        event_type=EventType.ENGAGEMENT_CANDIDATE,
        start_sec=start,
        end_sec=end,
        confidence=.9,
        evidence=[Evidence(
            frame_index=600,
            timestamp_sec=start,
            source="CS2FactionDetector.opposing_faction",
        )],
        attributes={
            "round_id": "round_001",
            "engagement_level": level,
            "first_shot_candidate_sec": shot,
        },
    )


def _round():
    return RoundAnalysis(round_id="round_001", start_sec=0, end_sec=60)


def test_fuses_native_highlight_with_firefight():
    result = detect_native_kills(
        [_round()],
        [_sample(22.0, highlighted=True, score=.82)],
        [_engagement()],
    )

    assert result.available
    assert result.highlight_episodes == 1
    assert result.matched_episodes == 1
    assert result.events[0].event_type == EventType.PLAYER_KILL
    assert result.events[0].attributes["classification"] == (
        "native_personal_kill"
    )
    assert len(result.events[0].evidence) == 2
    assert result.events[0].attributes["engagement_corroborated"] is True


def test_native_highlight_without_enemy_firefight_still_counts():
    result = detect_native_kills(
        [_round()],
        [
            _sample(22.0, highlighted=True, score=.9),
            _sample(22.5, highlighted=True, score=.85),
        ],
        [_engagement(level="visual_contact")],
    )

    assert result.available
    assert result.highlight_episodes == 1
    assert len(result.events) == 1
    assert result.events[0].attributes["engagement_corroborated"] is False


def test_single_frame_uncorroborated_outline_is_not_an_exact_kill():
    result = detect_native_kills(
        [_round()],
        [_sample(22.0, highlighted=True, score=.94)],
        [],
    )

    assert result.highlight_episodes == 1
    assert not result.events
    assert not result.available


def test_sparse_sampling_retains_exceptionally_strong_native_outline():
    result = detect_native_kills(
        [_round()],
        [_sample(22.0, highlighted=True, score=.99)],
        [],
    )

    assert len(result.events) == 1
    assert result.events[0].attributes["single_sample_strong_geometry"] is True


def test_firefight_without_native_highlight_is_not_a_kill():
    result = detect_native_kills(
        [_round()],
        [_sample(22.0)],
        [_engagement()],
    )

    assert not result.available
    assert result.highlight_episodes == 0
    assert not result.events


def test_contiguous_highlight_samples_form_one_kill():
    result = detect_native_kills(
        [_round()],
        [
            _sample(22.0, highlighted=True, score=.55),
            _sample(22.5, highlighted=True, score=.85),
            _sample(23.0, highlighted=True, score=.7),
        ],
        [_engagement()],
    )

    assert result.highlight_episodes == 1
    assert len(result.events) == 1
    assert result.events[0].start_sec == 22.0


def test_highlight_inside_round_does_not_require_sparse_engagement_sample():
    result = detect_native_kills(
        [_round()],
        [
            _sample(40.0, highlighted=True, score=.9),
            _sample(40.5, highlighted=True, score=.85),
        ],
        [_engagement()],
    )

    assert result.highlight_episodes == 1
    assert len(result.events) == 1


def test_same_feed_row_after_detector_gap_is_counted_once():
    result = detect_native_kills(
        [_round()],
        [
            _sample(22.0, highlighted=True, score=.8,
                    fingerprints=(b"same-row",)),
            _sample(26.0, highlighted=True, score=.9,
                    fingerprints=(b"same-row",)),
        ],
        [],
    )

    assert result.highlight_episodes == 2
    assert len(result.events) == 1
    assert result.events[0].start_sec == 22.0


def test_different_feed_rows_inside_merge_window_are_separate_kills():
    result = detect_native_kills(
        [_round()],
        [
            _sample(22.0, highlighted=True, score=.8,
                    fingerprints=(b"first-row",)),
            _sample(22.5, highlighted=True, score=.9,
                    fingerprints=(b"first-row", b"second-row")),
            _sample(23.0, highlighted=True, score=.85,
                    fingerprints=(b"first-row", b"second-row")),
        ],
        [],
    )

    assert result.highlight_episodes == 1
    assert len(result.events) == 2


def test_two_simultaneous_highlighted_rows_cannot_reuse_one_track():
    result = detect_native_kills(
        [_round()],
        [
            _sample(
                22.0,
                highlighted=True,
                score=.9,
                fingerprints=(b"first-row", b"second-row"),
            ),
            _sample(
                22.5,
                highlighted=True,
                score=.85,
                fingerprints=(b"first-row", b"second-row"),
            ),
        ],
        [],
    )

    assert len(result.events) == 2


def test_repeated_lower_row_insertions_expand_continuous_multikill():
    result = detect_native_kills(
        [_round()],
        [
            _sample(20.0, highlighted=True, score=.9,
                    fingerprints=(b"first",), positions=(.32,)),
            _sample(20.5, highlighted=True, score=.9,
                    fingerprints=(b"first", b"second"), positions=(.32, .45)),
            _sample(21.0, highlighted=True, score=.9,
                    fingerprints=(b"second",), positions=(.32,)),
            _sample(21.5, highlighted=True, score=.9,
                    fingerprints=(b"third", b"fourth"), positions=(.32, .46)),
            _sample(22.0, highlighted=True, score=.9,
                    fingerprints=(b"third", b"fourth"), positions=(.32, .46)),
        ],
        [],
    )

    assert len(result.events) == 3


def test_kill_is_retained_in_open_final_round():
    result = detect_native_kills(
        [RoundAnalysis(round_id="round_018", start_sec=1112.5, end_sec=None)],
        [_sample(
            1120.5,
            highlighted=True,
            score=.9,
            fingerprints=(b"final-round-awp-row",),
        ), _sample(
            1121.0,
            highlighted=True,
            score=.85,
            fingerprints=(b"final-round-awp-row",),
        )],
        [],
    )

    assert len(result.events) == 1
    assert result.events[0].attributes["round_id"] == "round_018"
    assert result.events[0].start_sec == 1120.5
