"""Concrete analysis pipeline wiring ingestion through to HUD parsing and object detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from gamesight.domain.models import AnalysisResult, Detection, HudState, VideoInput
from gamesight.ingestion.video_reader import VideoReader
from gamesight.perception.classifier import PlayerClassifier
from gamesight.perception.detector import ObjectDetector
from gamesight.perception.hud_parser import HudParser


class AnalysisPipeline(ABC):
    @abstractmethod
    def run(self, video: VideoInput) -> AnalysisResult:
        """Run the configured analysis workflow for one CS2 recording."""


class VideoAnalysisPipeline(AnalysisPipeline):
    """End-to-end pipeline: ingest video, sample frames, parse HUD state,
    and optionally run object detection with player classification.

    Dependency injection via *reader*, *parser*, *detector*, and
    *classifier* keeps the pipeline testable without real video files,
    OpenCV, or YOLO.

    Frame-level HUD states and detections are collected and reported as
    a structured summary inside ``AnalysisResult.warnings`` until the
    Event Engine can consume them to produce proper round/event output.
    """

    def __init__(
        self,
        reader: VideoReader,
        parser: HudParser,
        sample_fps: float = 10,
        detector: ObjectDetector | None = None,
        classifier: PlayerClassifier | None = None,
    ) -> None:
        self._reader = reader
        self._parser = parser
        self._sample_fps = sample_fps
        self._detector = detector
        self._classifier = classifier

    def run(self, video: VideoInput) -> AnalysisResult:
        metadata = self._reader.inspect(video)
        hud_states: list[HudState] = []
        all_detections: list[Detection] = []
        warnings: list[str] = []

        try:
            for frame in self._reader.frames(video, self._sample_fps):
                # HUD parsing
                state = self._parser.parse(
                    frame.image, frame.frame_index, frame.timestamp_sec
                )
                hud_states.append(state)

                # Object detection (optional)
                if self._detector is not None:
                    dets = self._detector.detect(
                        frame.image, frame.frame_index, frame.timestamp_sec
                    )
                    if self._classifier is not None and dets:
                        dets = self._classifier.classify(frame.image, dets)
                    all_detections.extend(dets)
        except Exception as exc:
            warnings.append(f"Frame processing error: {exc}")

        summary = _build_summary(hud_states, all_detections)
        for line in summary:
            warnings.append(line)

        return AnalysisResult(
            video=video,
            metadata=metadata,
            rounds=[],
            warnings=warnings,
        )


def _build_summary(
    states: list[HudState],
    detections: Sequence[Detection] | None = None,
) -> list[str]:
    """Build human-readable summary lines from collected HUD states and detections."""
    lines: list[str] = []

    total = len(states)
    lines.append(f"[pipeline] {total} frames processed")

    if detections is not None:
        total_dets = len(detections)
        lines.append(f"[pipeline] {total_dets} total detections across {total} frames")

        # Per-label counts
        label_counts: dict[str, int] = {}
        for d in detections:
            label_counts[d.label] = label_counts.get(d.label, 0) + 1
        for label, count in sorted(label_counts.items()):
            lines.append(f"[pipeline] detections.{label}: {count}")

        # Average confidence
        if total_dets > 0:
            avg_conf = sum(d.confidence for d in detections) / total_dets
            lines.append(f"[pipeline] detections.avg_confidence: {avg_conf:.2f}")

    if not states:
        return lines

    # HUD boolean flags
    flags: dict[str, int] = {}
    numeric: dict[str, list[float]] = {}

    for state in states:
        for key, val in state.values.items():
            if isinstance(val, bool):
                flags[key] = flags.get(key, 0) + (1 if val else 0)
            elif isinstance(val, (int, float)):
                numeric.setdefault(key, []).append(float(val))

    for key, count in sorted(flags.items()):
        pct = round(100 * count / max(total, 1))
        lines.append(f"[pipeline] {key}: {count}/{total} ({pct}%)")

    for key, values in sorted(numeric.items()):
        if not values:
            continue
        lines.append(
            f"[pipeline] {key}: min={min(values):.0f} "
            f"max={max(values):.0f} mean={sum(values)/len(values):.1f}"
        )

    return lines