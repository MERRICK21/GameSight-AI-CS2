"""Screenshot extractor — capture video frames at event timestamps.

Uses OpenCV to seek to specific frame indices and save screenshots
as PNG files.  Designed for dependency injection so tests can swap
in a mock extractor.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from importlib import import_module
from pathlib import Path

from gamesight.domain.models import AnalysisResult, EventType, Evidence, GameEvent
from gamesight.evidence.models import EvidenceClip, EvidenceImage


class ScreenshotExtractor(ABC):
    """Abstract interface for extracting screenshots from video."""

    @abstractmethod
    def extract(
        self,
        video_path: str | Path,
        events: Sequence[GameEvent],
        output_dir: Path | None = None,
    ) -> list[EvidenceImage]:
        """Extract screenshots for *events* and return EvidenceImage references."""


class OpenCVScreenshotExtractor(ScreenshotExtractor):
    """Extract screenshots using OpenCV's VideoCapture.

    For each event, seeks to the evidence frame (or nearest keyframe)
    and saves a PNG screenshot.

    Parameters
    ----------
    cv2_module:
        Inject a mock ``cv2`` for testing; ``None`` imports the real module.
    max_screenshots:
        Upper bound on screenshots extracted (default 50).  Events beyond
        this limit are silently skipped.
    """

    def __init__(
        self,
        cv2_module: object | None = None,
        max_screenshots: int = 50,
    ) -> None:
        self._cv2 = cv2_module if cv2_module is not None else import_module("cv2")
        self._max = max_screenshots

    def extract(
        self,
        video_path: str | Path,
        events: Sequence[GameEvent],
        output_dir: Path | None = None,
    ) -> list[EvidenceImage]:
        video_path = Path(video_path)
        out_dir = output_dir or Path(tempfile.mkdtemp(prefix="gamesight_evidence_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        capture = self._cv2.VideoCapture(str(video_path))
        images: list[EvidenceImage] = []

        try:
            if not capture.isOpened():
                return images

            width = int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))

            candidates = [
                (frame_idx, event)
                for event in events
                if (frame_idx := self._best_frame(event)) is not None
            ]
            candidates.sort(key=lambda item: item[0])
            current_frame: int | None = None
            cached_frame = None
            sequential = hasattr(capture, "grab")

            for frame_idx, event in candidates[:self._max]:
                if frame_idx == current_frame and cached_frame is not None:
                    success, frame = True, cached_frame
                elif sequential and current_frame is not None and frame_idx > current_frame:
                    success = True
                    # read() advances one frame; grab only the intervening
                    # frames, then decode the requested target once.
                    for _ in range(max(0, frame_idx - current_frame - 1)):
                        if not capture.grab():
                            success = False
                            break
                    if success:
                        success, frame = capture.read()
                    else:
                        frame = None
                else:
                    capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = capture.read()
                if not success:
                    continue
                current_frame = frame_idx
                cached_frame = frame

                img_id = f"evidence_{event.event_id}"
                img_path = out_dir / f"{img_id}.png"
                self._cv2.imwrite(str(img_path), frame)

                images.append(EvidenceImage(
                    image_id=img_id,
                    event_id=event.event_id,
                    frame_index=frame_idx,
                    timestamp_sec=event.start_sec,
                    image_path=str(img_path.resolve()),
                    source=event.evidence[0].source if event.evidence else "unknown",
                    width=width if width > 0 else None,
                    height=height if height > 0 else None,
                ))

        finally:
            capture.release()

        return images

    @staticmethod
    def _best_frame(event: GameEvent) -> int | None:
        """Pick the best frame index from an event's evidence list."""
        for ev in event.evidence:
            if ev.frame_index is not None:
                return ev.frame_index
        # Fall back to timestamp * 10 (approximate for 30fps video)
        return int(event.start_sec * 30)


class EvidenceClipExtractor:
    """Encode short browser-compatible H.264 clips around verified events."""

    def __init__(
        self,
        cv2_module: object | None = None,
        before_sec: float = 2.0,
        after_sec: float = 3.0,
        max_clips: int = 12,
        output_fps: float = 15.0,
        max_width: int = 960,
    ) -> None:
        self._cv2 = cv2_module if cv2_module is not None else import_module("cv2")
        self._before = max(0.0, before_sec)
        self._after = max(0.0, after_sec)
        self._max = max(0, max_clips)
        self._output_fps = max(1.0, output_fps)
        self._max_width = max(2, max_width)

    def extract(
        self,
        video_path: str | Path,
        events: Sequence[GameEvent],
        output_dir: Path | None = None,
    ) -> list[EvidenceClip]:
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "Evidence clips require PyAV. Install with: pip install av"
            ) from exc

        out_dir = output_dir or Path(tempfile.mkdtemp(prefix="gamesight_clips_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        capture = self._cv2.VideoCapture(str(video_path))
        clips: list[EvidenceClip] = []
        try:
            if not capture.isOpened() or self._max == 0:
                return clips
            native_fps = float(capture.get(self._cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(self._cv2.CAP_PROP_FRAME_COUNT))
            source_width = int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            if native_fps <= 0 or frame_count <= 0 or source_width <= 0 or source_height <= 0:
                return clips
            duration = frame_count / native_fps
            scale = min(1.0, self._max_width / source_width)
            width = max(2, int(source_width * scale) // 2 * 2)
            height = max(2, int(source_height * scale) // 2 * 2)
            target_fps = min(native_fps, self._output_fps)
            frame_step = max(1, int(round(native_fps / target_fps)))
            actual_fps = native_fps / frame_step

            seen: set[str] = set()
            selected: list[GameEvent] = []
            for event in events:
                if event.event_id not in seen:
                    selected.append(event)
                    seen.add(event.event_id)
                if len(selected) >= self._max:
                    break

            for event in selected:
                trigger = float(event.attributes.get("clip_trigger_sec", event.start_sec))
                start_sec = max(0.0, trigger - self._before)
                end_sec = min(duration, trigger + self._after)
                if end_sec <= start_sec:
                    continue
                clip_path = out_dir / f"clip_{event.event_id}.mp4"
                start_frame = max(0, int(start_sec * native_fps))
                end_frame = min(frame_count, int(end_sec * native_fps))
                capture.set(self._cv2.CAP_PROP_POS_FRAMES, start_frame)
                container = av.open(
                    str(clip_path), mode="w", options={"movflags": "+faststart"},
                )
                stream = container.add_stream(
                    "libx264", rate=Fraction(actual_fps).limit_denominator(1000),
                )
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "25", "preset": "veryfast"}
                encoded = 0
                try:
                    for frame_index in range(start_frame, end_frame):
                        success, image = capture.read()
                        if not success:
                            break
                        if (frame_index - start_frame) % frame_step:
                            continue
                        if image.shape[1] != width or image.shape[0] != height:
                            image = self._cv2.resize(
                                image, (width, height),
                                interpolation=self._cv2.INTER_AREA,
                            )
                        video_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                        for packet in stream.encode(video_frame):
                            container.mux(packet)
                        encoded += 1
                    for packet in stream.encode():
                        container.mux(packet)
                finally:
                    container.close()
                if encoded == 0 or not clip_path.is_file():
                    clip_path.unlink(missing_ok=True)
                    continue
                clips.append(EvidenceClip(
                    clip_id=f"clip_{event.event_id}",
                    event_id=event.event_id,
                    trigger_sec=trigger,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    video_path=str(clip_path.resolve()),
                    source=(event.evidence[0].source if event.evidence else "unknown"),
                    width=width,
                    height=height,
                    fps=round(actual_fps, 3),
                ))
        finally:
            capture.release()
        return clips


def build_round_keyframe_events(
    analysis: AnalysisResult,
    samples_per_round: int = 2,
    max_events: int = 30,
) -> list[GameEvent]:
    """Create evenly spaced representative-frame events for every round.

    These synthetic events are used only for screenshot extraction and live
    frame inspection.  They do not enter the match timeline or combat stats.
    Sampling inside each round avoids freeze-time frames at the exact boundary.
    """
    if samples_per_round < 1 or max_events < 1:
        return []

    fps = analysis.metadata.fps or 30.0
    video_end = analysis.metadata.duration_sec
    events: list[GameEvent] = []
    seen_frames: set[int] = set()

    for round_analysis in analysis.rounds:
        end_sec = round_analysis.end_sec
        if end_sec is None:
            end_sec = video_end
        if end_sec is None or end_sec <= round_analysis.start_sec:
            continue

        duration = end_sec - round_analysis.start_sec
        for sample_index in range(1, samples_per_round + 1):
            fraction = sample_index / (samples_per_round + 1)
            timestamp = round_analysis.start_sec + duration * fraction
            frame_index = max(0, int(round(timestamp * fps)))
            if frame_index in seen_frames:
                continue
            seen_frames.add(frame_index)

            event_id = (
                f"round_keyframe_{round_analysis.round_id}_{sample_index:02d}"
            )
            events.append(GameEvent(
                event_id=event_id,
                event_type=EventType.KEYFRAME,
                start_sec=timestamp,
                confidence=0.9,
                evidence=[Evidence(
                    frame_index=frame_index,
                    timestamp_sec=timestamp,
                    source="RoundKeyframeSampler",
                )],
                attributes={
                    "round_id": round_analysis.round_id,
                    "sample_index": sample_index,
                },
            ))
            if len(events) >= max_events:
                return events

    return events
