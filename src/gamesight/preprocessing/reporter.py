"""Aggregate ingestion outputs into a structured quality report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from gamesight.domain.models import VideoInput, VideoMetadata
from gamesight.preprocessing.normalizer import QualityDiagnostic
from gamesight.preprocessing.validator import ValidationResult


@dataclass
class IngestionReport:
    """Complete ingestion result for one CS2 recording.

    Aggregates every output produced during Sprint 1: metadata, quality
    diagnostics, validation outcome, and a sampling plan derived from the
    video container properties.
    """

    schema_version: str = "1.0"
    video_id: str = ""
    video_path: str = ""
    generated_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    quality: dict[str, object] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)
    sampling_plan: dict[str, object] = field(default_factory=dict)
    is_ready: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "video_path": self.video_path,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "quality": self.quality,
            "validation": self.validation,
            "sampling_plan": self.sampling_plan,
            "is_ready": self.is_ready,
            "summary": self.summary,
        }


class QualityReporter:
    """Combine ingestion artifacts into a single evidence-based report.

    The reporter is intentionally pure: it reads structured inputs and
    produces a structured output.  No inference, no heuristics beyond
    what is already encoded in the upstream modules.
    """

    def __init__(self, sample_fps: float = 10.0) -> None:
        if sample_fps <= 0:
            raise ValueError(f"sample_fps must be positive, got {sample_fps}")
        self._sample_fps = sample_fps

    # -- public API ----------------------------------------------------

    def generate(
        self,
        video: VideoInput,
        metadata: VideoMetadata,
        quality: QualityDiagnostic,
        validation: ValidationResult,
    ) -> IngestionReport:
        """Produce the final ingestion report from all upstream results."""
        sampling_plan = self._build_sampling_plan(metadata.fps, metadata.duration_sec)
        is_ready = validation.is_valid and self._quality_acceptable(quality)
        summary = self._build_summary(validation, quality, sampling_plan, is_ready)

        return IngestionReport(
            schema_version="1.0",
            video_id=video.video_id,
            video_path=str(video.path),
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata=self._metadata_dict(metadata),
            quality=quality.to_dict(),
            validation=validation.to_dict(),
            sampling_plan=sampling_plan,
            is_ready=is_ready,
            summary=summary,
        )

    # -- internal ------------------------------------------------------

    def _build_sampling_plan(
        self, native_fps: float | None, duration_sec: float | None
    ) -> dict[str, object]:
        plan: dict[str, object] = {
            "sample_fps": self._sample_fps,
            "native_fps": native_fps,
            "step": None,
            "estimated_frame_count": None,
            "estimated_duration_sec": duration_sec,
        }

        if native_fps is not None and native_fps > 0:
            step = max(1, round(native_fps / self._sample_fps))
            plan["step"] = step

            if duration_sec is not None and duration_sec > 0:
                plan["estimated_frame_count"] = int(
                    native_fps * duration_sec / step
                )

        return plan

    @staticmethod
    def _metadata_dict(metadata: VideoMetadata) -> dict[str, object]:
        return {
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "codec": metadata.codec,
            "duration_sec": metadata.duration_sec,
        }

    @staticmethod
    def _quality_acceptable(quality: QualityDiagnostic) -> bool:
        """A video is quality-acceptable when the core metrics are known.

        Unknown resolution or FPS blocks the pipeline downstream, so we
        require at least those to be present.
        """
        return quality.resolution_match or (
            quality.width is not None and quality.height is not None
        )

    @staticmethod
    def _build_summary(
        validation: ValidationResult,
        quality: QualityDiagnostic,
        sampling_plan: dict[str, object],
        is_ready: bool,
    ) -> str:
        parts: list[str] = []

        if is_ready:
            parts.append("Video passed ingestion checks.")
        else:
            reasons: list[str] = []
            reasons.extend(validation.errors)
            reasons.extend(validation.warnings)
            if not reasons:
                reasons.append("Quality assessment indicates issues.")
            parts.append(
                f"Video requires attention: {'; '.join(reasons)}"
            )

        est = sampling_plan.get("estimated_frame_count")
        if est is not None:
            parts.append(
                f"Estimated {est} frames at {sampling_plan['sample_fps']} fps."
            )

        return " ".join(parts)
