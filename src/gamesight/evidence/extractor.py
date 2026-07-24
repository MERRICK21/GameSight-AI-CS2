"""Screenshot extractor — capture video frames at event timestamps.

Uses OpenCV to seek to specific frame indices and save screenshots
as PNG files.  Designed for dependency injection so tests can swap
in a mock extractor.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from gamesight.domain.models import Evidence, GameEvent
from gamesight.evidence.models import EvidenceImage


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

            for event in events:
                if len(images) >= self._max:
                    break

                frame_idx = self._best_frame(event)
                if frame_idx is None:
                    continue

                capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = capture.read()
                if not success:
                    continue

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
