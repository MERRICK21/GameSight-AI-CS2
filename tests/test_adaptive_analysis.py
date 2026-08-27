from types import SimpleNamespace

from gamesight.orchestration.adaptive import (
    TimeWindow, build_refinement_windows, merge_samples, merge_time_windows,
)


def test_candidate_signals_build_clamped_merged_windows() -> None:
    samples = [
        SimpleNamespace(
            frame_index=10, timestamp_sec=1.0, flashed=True, scoped=False,
            shot_candidate=False, damage_candidate=False,
            local_kill_highlight=False, health_hud_visible=True,
        ),
        SimpleNamespace(
            frame_index=20, timestamp_sec=2.0, flashed=False, scoped=False,
            shot_candidate=True, damage_candidate=False,
            local_kill_highlight=False, health_hud_visible=True,
        ),
        SimpleNamespace(
            frame_index=90, timestamp_sec=9.0, flashed=False, scoped=False,
            shot_candidate=False, damage_candidate=False,
            local_kill_highlight=False, health_hud_visible=False,
        ),
    ]

    windows = build_refinement_windows(samples, duration_sec=10.0)

    assert windows == [TimeWindow(0.0, 6.0), TimeWindow(7.0, 10.0)]


def test_merge_windows_and_refined_samples_are_deterministic() -> None:
    assert merge_time_windows([
        TimeWindow(3.0, 4.0), TimeWindow(1.0, 2.5), TimeWindow(2.8, 3.2),
    ]) == [TimeWindow(1.0, 4.0)]
    base = [
        SimpleNamespace(frame_index=0, value="base-0"),
        SimpleNamespace(frame_index=10, value="base-10"),
    ]
    refined = [
        SimpleNamespace(frame_index=5, value="refined-5"),
        SimpleNamespace(frame_index=10, value="refined-10"),
    ]

    merged = merge_samples(base, refined)

    assert [item.frame_index for item in merged] == [0, 5, 10]
    assert merged[-1].value == "refined-10"
