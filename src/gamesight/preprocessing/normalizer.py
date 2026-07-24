"""Resolution, aspect-ratio, and quality normalization for ingested video."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from gamesight.domain.models import VideoInput, VideoMetadata


class Preprocessor(ABC):
    @abstractmethod
    def assess_quality(
        self, video: VideoInput, metadata: VideoMetadata
    ) -> dict[str, object]:
        """Return quality and domain-shift diagnostics for the source video."""

    @abstractmethod
    def normalize(
        self, video: VideoInput, metadata: VideoMetadata
    ) -> VideoInput:
        """Return a normalized analysis copy or a reference to the original input."""


@dataclass
class QualityDiagnostic:
    """Structured quality assessment for one ingested video.

    All match flags use a small tolerance so that container rounding
    (e.g. 1918 instead of 1920) does not produce false negatives.
    """

    width: int | None
    height: int | None
    fps: float | None
    target_width: int
    target_height: int
    target_fps: float
    resolution_match: bool
    aspect_ratio_match: bool
    fps_match: bool
    letterbox_needed: bool
    pillarbox_needed: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target_fps": self.target_fps,
            "resolution_match": self.resolution_match,
            "aspect_ratio_match": self.aspect_ratio_match,
            "fps_match": self.fps_match,
            "letterbox_needed": self.letterbox_needed,
            "pillarbox_needed": self.pillarbox_needed,
            "warnings": self.warnings,
        }


class VideoPreprocessor(Preprocessor):
    """Inspect video metadata against a target specification.

    Compares resolution, aspect ratio, and frame rate to expected norms.
    The *normalize* step currently returns the original input unchanged;
    actual re-encoding is deferred to a future sprint.
    """

    _ASPECT_TOLERANCE = 0.02
    _RESOLUTION_TOLERANCE = 4
    _FPS_TOLERANCE = 0.5

    def __init__(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: float = 60.0,
    ) -> None:
        self._target_width = target_width
        self._target_height = target_height
        self._target_fps = target_fps

    # -- Preprocessor interface ----------------------------------------

    def assess_quality(
        self, video: VideoInput, metadata: VideoMetadata
    ) -> dict[str, object]:
        """Run all quality checks and return a serialisable diagnostic."""
        return self._build_diagnostic(metadata).to_dict()

    def normalize(
        self, video: VideoInput, metadata: VideoMetadata
    ) -> VideoInput:
        """Return the input unchanged (transcoding not yet implemented)."""
        return video

    # -- internal ------------------------------------------------------

    def _build_diagnostic(self, metadata: VideoMetadata) -> QualityDiagnostic:
        width = metadata.width
        height = metadata.height
        fps = metadata.fps
        warnings: list[str] = []

        resolution_match = self._check_resolution(width, height, warnings)
        aspect_ratio_match = self._check_aspect_ratio(
            width, height, warnings
        )
        fps_match = self._check_fps(fps, warnings)
        letterbox_needed, pillarbox_needed = self._check_letterbox(
            width, height
        )

        return QualityDiagnostic(
            width=width,
            height=height,
            fps=fps,
            target_width=self._target_width,
            target_height=self._target_height,
            target_fps=self._target_fps,
            resolution_match=resolution_match,
            aspect_ratio_match=aspect_ratio_match,
            fps_match=fps_match,
            letterbox_needed=letterbox_needed,
            pillarbox_needed=pillarbox_needed,
            warnings=warnings,
        )

    def _check_resolution(
        self, width: int | None, height: int | None, warnings: list[str]
    ) -> bool:
        if width is None or height is None:
            warnings.append("Video resolution is unknown.")
            return False

        w_ok = abs(width - self._target_width) <= self._RESOLUTION_TOLERANCE
        h_ok = abs(height - self._target_height) <= self._RESOLUTION_TOLERANCE

        if not w_ok or not h_ok:
            warnings.append(
                f"Resolution {width}x{height} differs from "
                f"target {self._target_width}x{self._target_height}."
            )
            return False
        return True

    def _check_aspect_ratio(
        self, width: int | None, height: int | None, warnings: list[str]
    ) -> bool:
        if width is None or height is None or height == 0:
            return True  # cannot assess; silence duplicate warning

        target_ratio = self._target_width / self._target_height
        actual_ratio = width / height

        if abs(actual_ratio - target_ratio) > self._ASPECT_TOLERANCE:
            warnings.append(
                f"Aspect ratio {actual_ratio:.3f} differs from "
                f"target {target_ratio:.3f} (16:9)."
            )
            return False
        return True

    def _check_fps(
        self, fps: float | None, warnings: list[str]
    ) -> bool:
        if fps is None:
            warnings.append("Frame rate is unknown.")
            return False

        if abs(fps - self._target_fps) > self._FPS_TOLERANCE:
            warnings.append(
                f"Frame rate {fps} fps differs from target {self._target_fps}."
            )
            return False
        return True

    def _check_letterbox(
        self, width: int | None, height: int | None
    ) -> tuple[bool, bool]:
        if width is None or height is None:
            return False, False

        target_ratio = self._target_width / self._target_height
        actual_ratio = width / height

        letterbox = actual_ratio > target_ratio + self._ASPECT_TOLERANCE
        pillarbox = actual_ratio < target_ratio - self._ASPECT_TOLERANCE
        return letterbox, pillarbox
