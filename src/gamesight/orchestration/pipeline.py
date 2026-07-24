"""Concrete analysis pipeline wiring ingestion through to HUD parsing, object detection, and tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from gamesight.domain.models import AnalysisResult, Detection, HudState, Track, VideoInput
from gamesight.ingestion.video_reader import VideoReader
from gamesight.perception.classifier import PlayerClassifier
from gamesight.perception.detector import ObjectDetector
from gamesight.perception.hud_parser import HudParser
from gamesight.tracking.tracker import MultiObjectTracker


class AnalysisPipeline(ABC):
    @abstractmethod
    def run(self, video: VideoInput) -> AnalysisResult:
        """Run the configured analysis workflow for one CS2 recording."""


class VideoAnalysisPipeline(AnalysisPipeline):
    """End-to-end pipeline: ingest, parse HUD, detect, classify, track.

    Dependency injection via *reader*, *parser*, *detector*,
    *classifier*, and *tracker* keeps the pipeline testable without
    real video files, OpenCV, or YOLO.
    """

    def __init__(
        self,
        reader: VideoReader,
        parser: HudParser,
        sample_fps: float = 10,
        detector: ObjectDetector | None = None,
        classifier: PlayerClassifier | None = None,
        tracker: MultiObjectTracker | None = None,
    ) -> None:
        self._reader = reader
        self._parser = parser
        self._sample_fps = sample_fps
        self._detector = detector
        self._classifier = classifier
        self._tracker = tracker

    def run(self, video: VideoInput) -> AnalysisResult:
        metadata = self._reader.inspect(video)
        hud_states: list[HudState] = []
        all_detections: list[Detection] = []
        all_tracks: list[Track] = []
        warnings: list[str] = []

        try:
            for frame in self._reader.frames(video, self._sample_fps):
                # HUD parsing
                state = self._parser.parse(
                    frame.image, frame.frame_index, frame.timestamp_sec
                )
                hud_states.append(state)

                # Object detection + classification
                if self._detector is not None:
                    dets = self._detector.detect(
                        frame.image, frame.frame_index, frame.timestamp_sec
                    )
                    if self._classifier is not None and dets:
                        dets = self._classifier.classify(frame.image, dets)
                    all_detections.extend(dets)

                    # Tracking
                    if self._tracker is not None:
                        tracks = self._tracker.update(dets)
                        all_tracks.extend(tracks)
        except Exception as exc:
            warnings.append(f"Frame processing error: {exc}")

        summary = _build_summary(hud_states, all_detections, all_tracks)
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
    tracks: Sequence[Track] | None = None,
) -> list[str]:
    """Build human-readable summary lines from HUD, detections, and tracks."""
    lines: list[str] = []
    total = len(states)
    lines.append(f"[pipeline] {total} frames processed")

    if detections is not None:
        total_dets = len(detections)
        lines.append(f"[pipeline] {total_dets} total detections across {total} frames")
        label_counts: dict[str, int] = {}
        for d in detections:
            label_counts[d.label] = label_counts.get(d.label, 0) + 1
        for label, count in sorted(label_counts.items()):
            lines.append(f"[pipeline] detections.{label}: {count}")
        if total_dets > 0:
            avg_conf = sum(d.confidence for d in detections) / total_dets
            lines.append(f"[pipeline] detections.avg_confidence: {avg_conf:.2f}")

    if tracks is not None:
        # Deduplicate: tracker returns same tracks each frame; use final snapshot
        unique_tracks: dict[str, Track] = {}
        for t in tracks:
            unique_tracks[t.track_id] = t
        lines.append(f"[pipeline] {len(unique_tracks)} unique tracks")
        for tid in sorted(unique_tracks.keys()):
            t = unique_tracks[tid]
            lines.append(f"[pipeline] tracks.{tid}: label={t.label} detections={len(t.detections)}")

    if not states:
        return lines

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