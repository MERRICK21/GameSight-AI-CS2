"""Concrete analysis pipeline wiring ingestion through to HUD parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gamesight.domain.models import AnalysisResult, HudState, VideoInput
from gamesight.ingestion.video_reader import VideoReader
from gamesight.perception.hud_parser import HudParser


class AnalysisPipeline(ABC):
    @abstractmethod
    def run(self, video: VideoInput) -> AnalysisResult:
        """Run the configured analysis workflow for one CS2 recording."""


class VideoAnalysisPipeline(AnalysisPipeline):
    """End-to-end pipeline: ingest video, sample frames, parse HUD state.

    Dependency injection via *reader* and *parser* keeps the pipeline
    testable without real video files or OpenCV.

    Frame-level HUD states are collected and reported as a structured
    summary inside ``AnalysisResult.warnings`` until the Event Engine
    (Sprint 3) can consume them to produce proper round/event output.
    """

    def __init__(
        self,
        reader: VideoReader,
        parser: HudParser,
        sample_fps: float = 10,
    ) -> None:
        self._reader = reader
        self._parser = parser
        self._sample_fps = sample_fps

    def run(self, video: VideoInput) -> AnalysisResult:
        metadata = self._reader.inspect(video)
        hud_states: list[HudState] = []
        warnings: list[str] = []

        try:
            for frame in self._reader.frames(video, self._sample_fps):
                state = self._parser.parse(
                    frame.image, frame.frame_index, frame.timestamp_sec
                )
                hud_states.append(state)
        except Exception as exc:
            warnings.append(f"Frame processing error: {exc}")

        summary = _build_summary(hud_states)
        for line in summary:
            warnings.append(line)

        return AnalysisResult(
            video=video,
            metadata=metadata,
            rounds=[],
            warnings=warnings,
        )


def _build_summary(states: list[HudState]) -> list[str]:
    """Build human-readable summary lines from collected HUD states."""
    if not states:
        return ["[pipeline] 0 frames processed"]

    total = len(states)

    # Count boolean flags across all frames
    flags: dict[str, int] = {}
    numeric: dict[str, list[float]] = {}

    for state in states:
        for key, val in state.values.items():
            if isinstance(val, bool):
                flags[key] = flags.get(key, 0) + (1 if val else 0)
            elif isinstance(val, (int, float)):
                numeric.setdefault(key, []).append(float(val))

    lines = [f"[pipeline] {total} frames processed"]

    for key, count in sorted(flags.items()):
        pct = round(100 * count / total)
        lines.append(f"[pipeline] {key}: {count}/{total} ({pct}%)")

    for key, values in sorted(numeric.items()):
        if not values:
            continue
        lines.append(
            f"[pipeline] {key}: min={min(values):.0f} "
            f"max={max(values):.0f} mean={sum(values)/len(values):.1f}"
        )

    return lines
