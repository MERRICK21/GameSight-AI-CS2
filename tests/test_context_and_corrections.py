from pathlib import Path
from types import SimpleNamespace

from gamesight.coach.engine import RuleBasedCoach
from gamesight.domain.models import (
    AnalysisResult, ContextObservation, EventType, GameEvent, RoundAnalysis,
    RoundContextEvidence, VideoInput, VideoMetadata,
)
from gamesight.i18n.loader import I18nLoader
from gamesight.perception.context import build_round_contexts, context_coverage
from gamesight.reporting.corrections import (
    add_manual_events, apply_event_corrections, build_correction_export,
    diagnostic_rows,
)


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        video=VideoInput(
            video_id="match", path=Path("match.mp4"), source_name="test1.mp4",
        ),
        metadata=VideoMetadata(duration_sec=30.0, fps=30.0),
        rounds=[RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=30.0,
            events=[
                GameEvent(
                    event_id="kill-1", event_type=EventType.PLAYER_KILL,
                    start_sec=12.0, confidence=.8,
                    attributes={"method": "native"},
                ),
                GameEvent(
                    event_id="death-1", event_type=EventType.PLAYER_DEATH,
                    start_sec=25.0, confidence=.9,
                ),
            ],
        )],
    )


def test_context_builder_records_side_and_preserves_unknown_fields() -> None:
    contexts = build_round_contexts(_analysis().rounds, [
        SimpleNamespace(timestamp_sec=1.0, player_team="t"),
        SimpleNamespace(timestamp_sec=2.0, player_team="t"),
        SimpleNamespace(timestamp_sec=3.0, player_team=None),
    ])

    assert contexts[0].player_side == "t"
    assert contexts[0].player_side_confidence == 1.0
    assert contexts[0].weapon is None
    assert contexts[0].native_round_clock_sec is None
    assert context_coverage(contexts) == {
        "rounds": 1, "player_side": 1, "native_round_clock": 0,
        "weapon": 0, "economy": 0, "utility": 0, "map_position": 0,
    }


def test_context_builder_requires_continuous_decreasing_native_clock() -> None:
    observations = [
        ContextObservation(
            frame_index=0, timestamp_sec=1.0, value=115, confidence=.8,
            source="OCRRoundDetector.native_round_clock",
        ),
        ContextObservation(
            frame_index=60, timestamp_sec=3.0, value=113, confidence=.9,
            source="OCRRoundDetector.native_round_clock",
        ),
        ContextObservation(
            frame_index=120, timestamp_sec=5.0, value=45, confidence=.9,
            source="OCRRoundDetector.native_round_clock",
        ),
    ]

    context = build_round_contexts(
        _analysis().rounds, [], observations,
    )[0]

    assert context.native_round_clock_sec == 115
    assert [item.value for item in context.round_clock_observations] == [115, 113]


def test_context_builder_rejects_single_or_static_clock_read() -> None:
    observations = [
        {"timestamp_sec": 1.0, "value": 115, "confidence": .9,
         "source": "native"},
        {"timestamp_sec": 3.0, "value": 115, "confidence": .9,
         "source": "native"},
    ]

    context = build_round_contexts(_analysis().rounds, [], observations)[0]

    assert context.native_round_clock_sec is None


def test_context_builder_keeps_repeated_held_weapon_evidence() -> None:
    samples = [
        SimpleNamespace(
            frame_index=30, timestamp_sec=1.0, player_team="t",
            weapon="c4", weapon_confidence=.91,
            weapon_source="FirstPersonWeaponClassifier.held_c4_geometry",
        ),
        SimpleNamespace(
            frame_index=45, timestamp_sec=1.5, player_team="t",
            weapon="c4", weapon_confidence=.91,
            weapon_source="FirstPersonWeaponClassifier.held_c4_geometry",
        ),
    ]

    context = build_round_contexts(_analysis().rounds, samples)[0]

    assert context.weapon == "c4"
    assert context.weapon_categories == ["c4"]
    assert len(context.weapon_observations) == 2


def test_context_builder_verifies_money_and_native_mirage_position() -> None:
    money = [
        {"timestamp_sec": 4.0, "value": 2750, "confidence": .95,
         "source": "OCRRoundDetector.native_money"},
        {"timestamp_sec": 10.0, "value": 2750, "confidence": .90,
         "source": "OCRRoundDetector.native_money"},
    ]
    positions = [
        {"timestamp_sec": 6.0, "value": "Palace Interior",
         "confidence": .98,
         "source": "OCRRoundDetector.native_location_text"},
    ]

    context = build_round_contexts(
        _analysis().rounds, [], money_observations=money,
        position_observations=positions,
    )[0]

    assert context.money == 2750
    assert len(context.money_observations) == 2
    assert context.map_name == "mirage"
    assert context.map_position == "palace_interior"
    assert context.map_name_confidence == .95


def test_context_builder_rejects_single_money_and_unknown_location_text() -> None:
    context = build_round_contexts(
        _analysis().rounds, [],
        money_observations=[{
            "timestamp_sec": 4.0, "value": 2750, "confidence": .99,
            "source": "native",
        }],
        position_observations=[{
            "timestamp_sec": 6.0, "value": "Bombsite E",
            "confidence": .99, "source": "native",
        }],
    )[0]

    assert context.money is None
    assert context.map_name is None
    assert context.map_position is None


def test_context_builder_keeps_repeated_held_utility_evidence() -> None:
    samples = [
        SimpleNamespace(
            frame_index=30, timestamp_sec=1.0, player_team="t",
            held_utility="smoke", utility_confidence=.91,
            utility_source="test.held_view_model",
        ),
        SimpleNamespace(
            frame_index=45, timestamp_sec=1.5, player_team="t",
            held_utility="smoke", utility_confidence=.90,
            utility_source="test.held_view_model",
        ),
    ]

    context = build_round_contexts(_analysis().rounds, samples)[0]

    assert context.utility == ["smoke"]
    assert len(context.utility_observations) == 2


def test_context_guard_abstains_in_selected_language() -> None:
    coach = RuleBasedCoach(I18nLoader("zh-CN"))
    context = RoundContextEvidence(round_id="round_001", player_side="t")

    guarded = coach._with_context_guard("已建立遇敌复盘窗口。", context)

    assert "已知：玩家阵营" in guarded
    assert "原生回合计时" in guarded
    assert "不生成" in guarded


def test_rejected_detection_is_removed_without_mutating_raw_analysis() -> None:
    analysis = _analysis()

    corrected = apply_event_corrections(analysis, {"kill-1": False})

    assert [event.event_id for event in analysis.rounds[0].events] == [
        "kill-1", "death-1",
    ]
    assert [event.event_id for event in corrected.rounds[0].events] == [
        "death-1",
    ]
    assert corrected.analysis_metadata["manual_corrections"] == 1


def test_diagnostic_export_retains_raw_labels_and_acceptance() -> None:
    analysis = _analysis()
    rows = diagnostic_rows(analysis, {"kill-1": False})
    exported = build_correction_export(analysis, {"kill-1": False})

    assert rows[0]["accepted"] is False
    assert rows[1]["accepted"] is True
    assert exported["source_name"] == "test1.mp4"
    assert exported["labels"] == rows


def test_manual_missed_event_is_added_as_separate_user_label() -> None:
    analysis = _analysis()

    augmented = add_manual_events(analysis, [{
        "event_id": "manual-player-kill",
        "round_id": "round_001",
        "event_type": "player_kill",
        "timestamp_sec": 18.5,
    }])

    assert len(analysis.rounds[0].events) == 2
    assert len(augmented.rounds[0].events) == 3
    added = next(
        event for event in augmented.rounds[0].events
        if event.event_id == "manual-player-kill"
    )
    assert added.attributes["method"] == "manual_user_label"
    assert added.evidence[0].source == "UserCorrection.manual_label"
