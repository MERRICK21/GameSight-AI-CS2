"""Demo data generators and pipeline runner for the Streamlit app.

These functions are streamlit-free so they can be tested independently.
"""

from __future__ import annotations

from pathlib import Path

from gamesight.domain.models import (
    AnalysisResult,
    Detection,
    EventType,
    Evidence,
    GameEvent,
    RoundAnalysis,
    Track,
    VideoInput,
    VideoMetadata,
)
from gamesight.events.aggregator import aggregate_events
from gamesight.events.detectors import KillEventDetector, RoundBoundaryDetector


def generate_demo_events(rounds: int = 5) -> list[GameEvent]:
    """Generate synthetic CS2 events for demo mode."""
    events: list[GameEvent] = []
    t = 0.0
    for r in range(1, rounds + 1):
        rid = f"round_{r:03d}"
        # Round start
        events.append(GameEvent(
            event_id=f"round_start_{rid}",
            event_type=EventType.ROUND_START,
            start_sec=t,
            confidence=0.95,
            evidence=[Evidence(timestamp_sec=t, frame_index=int(t * 10), source="RoundBoundaryDetector")],
            attributes={"round_id": rid},
        ))
        t += 3.0  # freeze time

        # Enemy first visible
        events.append(GameEvent(
            event_id=f"enemy_first_visible_{rid}",
            event_type=EventType.ENEMY_FIRST_VISIBLE,
            start_sec=t + 5.0,
            confidence=0.80,
            evidence=[Evidence(timestamp_sec=t + 5.0, frame_index=int((t + 5) * 10), source="demo")],
        ))

        # Some kills
        n_kills = max(0, min(3, r % 4))
        for k in range(n_kills):
            events.append(GameEvent(
                event_id=f"player_kill_{rid}_{k}",
                event_type=EventType.PLAYER_KILL,
                start_sec=t + 8.0 + k * 5.0,
                confidence=0.55,
                evidence=[Evidence(timestamp_sec=t + 8.0 + k * 5, frame_index=int((t + 8 + k * 5) * 10), source="KillEventDetector")],
                attributes={"kill_index": k + 1},
            ))

        # Possibly a death
        if r % 3 == 0:
            events.append(GameEvent(
                event_id=f"player_death_{rid}",
                event_type=EventType.PLAYER_DEATH,
                start_sec=t + 18.0,
                confidence=0.85,
                evidence=[Evidence(timestamp_sec=t + 18.0, frame_index=int((t + 18) * 10), source="KillEventDetector")],
                attributes={"death_index": 1},
            ))

        # Round end
        round_dur = 90.0 + r * 5.0
        t += round_dur
        events.append(GameEvent(
            event_id=f"round_end_{rid}",
            event_type=EventType.ROUND_END,
            start_sec=t,
            confidence=0.90,
            evidence=[Evidence(timestamp_sec=t, frame_index=int(t * 10), source="RoundBoundaryDetector")],
            attributes={"round_id": rid},
        ))
        t += 10.0  # between rounds
    return events


def generate_demo_tracks() -> list[Track]:
    """Generate synthetic player tracks for demo mode."""
    return [
        Track(
            track_id="track_0000",
            label="enemy",
            detections=[
                Detection(label="enemy", confidence=0.88, bbox_xyxy=(400.0, 200.0, 480.0, 350.0), frame_index=50, timestamp_sec=5.0),
                Detection(label="enemy", confidence=0.85, bbox_xyxy=(420.0, 210.0, 500.0, 360.0), frame_index=60, timestamp_sec=6.0),
                Detection(label="enemy", confidence=0.90, bbox_xyxy=(440.0, 205.0, 510.0, 355.0), frame_index=70, timestamp_sec=7.0),
            ],
        ),
        Track(
            track_id="track_0001",
            label="enemy",
            detections=[
                Detection(label="enemy", confidence=0.82, bbox_xyxy=(800.0, 300.0, 880.0, 450.0), frame_index=120, timestamp_sec=12.0),
                Detection(label="enemy", confidence=0.79, bbox_xyxy=(810.0, 310.0, 890.0, 460.0), frame_index=130, timestamp_sec=13.0),
            ],
        ),
        Track(
            track_id="track_0002",
            label="teammate",
            detections=[
                Detection(label="teammate", confidence=0.91, bbox_xyxy=(200.0, 400.0, 280.0, 550.0), frame_index=80, timestamp_sec=8.0),
                Detection(label="teammate", confidence=0.93, bbox_xyxy=(210.0, 410.0, 290.0, 560.0), frame_index=90, timestamp_sec=9.0),
            ],
        ),
    ]


def run_demo_pipeline(
    video_path: str,
    sample_fps: float,
) -> tuple[AnalysisResult, list[Track]]:
    """Run the full analysis pipeline with demo/synthetic data.

    Returns (AnalysisResult, list[Track]) ready for timeline building and
    report generation.
    """
    demo_events = generate_demo_events(rounds=5)
    demo_tracks = generate_demo_tracks()

    # Event detection (simulated — demo events are already formed).
    rbd = RoundBoundaryDetector()
    ked = KillEventDetector()

    rounds = aggregate_events(demo_events)

    metadata = VideoMetadata(
        duration_sec=sum((r.end_sec or 0) - r.start_sec for r in rounds) + 100.0,
        fps=30.0,
        width=1920,
        height=1080,
        codec="h264",
    )

    analysis = AnalysisResult(
        video=VideoInput(video_id=Path(video_path).stem, path=Path(video_path)),
        metadata=metadata,
        rounds=rounds,
    )

    return analysis, demo_tracks
