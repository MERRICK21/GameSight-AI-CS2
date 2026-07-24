"""Hard-constraint video validation before pipeline processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gamesight.domain.models import VideoInput, VideoMetadata


@dataclass
class ValidationResult:
    """Outcome of video validation with errors (hard blocks) and warnings."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class VideoValidator:
    """Enforce minimum requirements before a video enters the pipeline.

    Checks are intentionally conservative: unknowns produce errors rather
    than false passes.  Warnings flag recoverable concerns that degrade
    quality but do not block processing.
    """

    _DEFAULT_EXTENSIONS = {".mp4", ".mov", ".mkv"}
    _MIN_WIDTH = 640
    _MIN_HEIGHT = 360
    _MIN_FPS = 1.0

    def __init__(
        self,
        accepted_extensions: set[str] | None = None,
        min_width: int = _MIN_WIDTH,
        min_height: int = _MIN_HEIGHT,
        min_fps: float = _MIN_FPS,
    ) -> None:
        self._extensions = accepted_extensions or self._DEFAULT_EXTENSIONS
        self._min_width = min_width
        self._min_height = min_height
        self._min_fps = min_fps

    def validate(
        self, video: VideoInput, metadata: VideoMetadata
    ) -> ValidationResult:
        """Run all validation checks and return a structured result."""
        errors: list[str] = []
        warnings: list[str] = []

        self._check_file_exists(video.path, errors)
        self._check_extension(video.path, errors)
        self._check_resolution(metadata, errors, warnings)
        self._check_frame_rate(metadata, errors, warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_file_exists(self, path: Path, errors: list[str]) -> None:
        if not path.exists():
            errors.append(f"Video file not found: {path}")

    def _check_extension(self, path: Path, errors: list[str]) -> None:
        suffix = path.suffix.lower()
        if suffix not in self._extensions:
            errors.append(
                f"Unsupported file extension '{suffix}'. "
                f"Accepted: {', '.join(sorted(self._extensions))}"
            )

    def _check_resolution(
        self,
        metadata: VideoMetadata,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if metadata.width is None or metadata.height is None:
            errors.append("Video resolution is unknown.")
            return

        if metadata.width < self._min_width or metadata.height < self._min_height:
            warnings.append(
                f"Resolution {metadata.width}x{metadata.height} is below "
                f"recommended minimum {self._min_width}x{self._min_height}."
            )

    def _check_frame_rate(
        self,
        metadata: VideoMetadata,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if metadata.fps is None:
            errors.append("Frame rate is unknown.")
            return

        if metadata.fps < self._min_fps:
            warnings.append(
                f"Frame rate {metadata.fps} fps is below "
                f"recommended minimum {self._min_fps} fps."
            )

        if metadata.fps > 240:
            warnings.append(
                f"Unusually high frame rate: {metadata.fps} fps."
            )
