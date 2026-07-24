"""Unit tests for ingestion quality report generation."""

from pathlib import Path
from unittest import TestCase

from gamesight.domain.models import VideoInput, VideoMetadata
from gamesight.preprocessing.normalizer import QualityDiagnostic
from gamesight.preprocessing.validator import ValidationResult
from gamesight.preprocessing.reporter import IngestionReport, QualityReporter


class IngestionReportTests(TestCase):
    """Tests for IngestionReport serialisation."""

    def test_to_dict_serialises_all_sections(self) -> None:
        report = IngestionReport(
            video_id="v1",
            video_path="/tmp/v1.mp4",
            generated_at="2026-07-24T00:00:00Z",
            metadata={"fps": 60.0},
            quality={"resolution_match": True},
            validation={"is_valid": True},
            sampling_plan={"sample_fps": 10.0, "step": 6},
            is_ready=True,
            summary="All clear.",
        )
        d = report.to_dict()

        self.assertEqual(d["schema_version"], "1.0")
        self.assertEqual(d["video_id"], "v1")
        self.assertEqual(d["is_ready"], True)
        self.assertEqual(d["summary"], "All clear.")
        self.assertIn("metadata", d)
        self.assertIn("quality", d)
        self.assertIn("validation", d)
        self.assertIn("sampling_plan", d)

    def test_default_report_is_empty_and_not_ready(self) -> None:
        report = IngestionReport()
        self.assertFalse(report.is_ready)
        self.assertEqual(report.summary, "")


class QualityReporterTests(TestCase):
    """Tests for QualityReporter.generate()."""

    def setUp(self) -> None:
        self.reporter = QualityReporter(sample_fps=10.0)

    def _video(self) -> VideoInput:
        return VideoInput(video_id="v1", path=Path("match.mp4"))

    def _meta(self, **kw: object) -> VideoMetadata:
        defaults: dict[str, object] = {
            "width": 1920, "height": 1080, "fps": 60.0,
            "duration_sec": 300.0, "codec": "avc1",
        }
        defaults.update(kw)
        return VideoMetadata(**defaults)  # type: ignore[arg-type]

    def _quality(self, **kw: object) -> QualityDiagnostic:
        defaults: dict[str, object] = {
            "width": 1920, "height": 1080, "fps": 60.0,
            "target_width": 1920, "target_height": 1080, "target_fps": 60.0,
            "resolution_match": True, "aspect_ratio_match": True,
            "fps_match": True,
            "letterbox_needed": False, "pillarbox_needed": False,
            "warnings": [],
        }
        defaults.update(kw)
        return QualityDiagnostic(**defaults)  # type: ignore[arg-type]

    def _validation(self, **kw: object) -> ValidationResult:
        defaults: dict[str, object] = {
            "is_valid": True, "errors": [], "warnings": [],
        }
        defaults.update(kw)
        return ValidationResult(**defaults)  # type: ignore[arg-type]

    # -- ready video ----------------------------------------------------

    def test_perfect_video_is_ready(self) -> None:
        report = self.reporter.generate(
            self._video(), self._meta(), self._quality(),
            self._validation(),
        )
        self.assertTrue(report.is_ready)
        self.assertIn("passed", report.summary.lower())

    # -- not ready ------------------------------------------------------

    def test_validation_failure_marks_not_ready(self) -> None:
        report = self.reporter.generate(
            self._video(), self._meta(), self._quality(),
            self._validation(is_valid=False, errors=["file not found"]),
        )
        self.assertFalse(report.is_ready)
        self.assertIn("requires attention", report.summary.lower())

    def test_missing_resolution_marks_not_ready(self) -> None:
        report = self.reporter.generate(
            self._video(),
            self._meta(),
            self._quality(
                width=None, height=None, resolution_match=False,
            ),
            self._validation(),
        )
        self.assertFalse(report.is_ready)

    def test_custom_quality_unacceptable_flags_not_ready(self) -> None:
        """Known-but-mismatched resolution is still acceptable."""
        report = self.reporter.generate(
            self._video(),
            self._meta(),
            self._quality(
                width=640, height=480, resolution_match=False,
                aspect_ratio_match=False,
            ),
            self._validation(),
        )
        # resolution is known (640x480), so still acceptable
        self.assertTrue(report.is_ready)

    def test_unknown_fps_does_not_block_if_resolution_ok(self) -> None:
        report = self.reporter.generate(
            self._video(),
            self._meta(fps=None),
            self._quality(fps=None, fps_match=False),
            self._validation(),
        )
        # Resolution is fine, so it should be ready
        # fps_match=False but width/height are known
        self.assertTrue(report.is_ready)

    # -- sampling plan --------------------------------------------------

    def test_sampling_plan_for_60fps_10fps_sample(self) -> None:
        report = self.reporter.generate(
            self._video(),
            self._meta(fps=60.0, duration_sec=300.0),
            self._quality(),
            self._validation(),
        )
        plan = report.sampling_plan

        self.assertEqual(plan["sample_fps"], 10.0)
        self.assertEqual(plan["native_fps"], 60.0)
        self.assertEqual(plan["step"], 6)
        self.assertEqual(plan["estimated_frame_count"], 3000)  # 60*300/6
        self.assertEqual(plan["estimated_duration_sec"], 300.0)

    def test_sampling_plan_handles_missing_fps(self) -> None:
        report = self.reporter.generate(
            self._video(),
            self._meta(fps=None, duration_sec=None),
            self._quality(fps=None, fps_match=False),
            self._validation(),
        )
        plan = report.sampling_plan

        self.assertIsNone(plan["native_fps"])
        self.assertIsNone(plan["step"])
        self.assertIsNone(plan["estimated_frame_count"])

    def test_sampling_plan_step_clamped_to_one(self) -> None:
        """sample_fps > native_fps should give step=1."""
        reporter = QualityReporter(sample_fps=120.0)
        report = reporter.generate(
            self._video(),
            self._meta(fps=30.0, duration_sec=10.0),
            self._quality(fps=30.0, fps_match=False),
            self._validation(),
        )
        self.assertEqual(report.sampling_plan["step"], 1)

    # -- report structure -----------------------------------------------

    def test_report_includes_all_sections(self) -> None:
        report = self.reporter.generate(
            self._video(), self._meta(), self._quality(),
            self._validation(),
        )

        self.assertIn("fps", report.metadata)
        self.assertIn("width", report.metadata)
        self.assertIn("resolution_match", report.quality)
        self.assertIn("is_valid", report.validation)
        self.assertIn("step", report.sampling_plan)
        self.assertIsNotNone(report.generated_at)

    def test_video_id_and_path_are_preserved(self) -> None:
        report = self.reporter.generate(
            self._video(), self._meta(), self._quality(),
            self._validation(),
        )
        self.assertEqual(report.video_id, "v1")
        self.assertEqual(report.video_path, "match.mp4")

    # -- error on invalid sample_fps ------------------------------------

    def test_raises_on_non_positive_sample_fps(self) -> None:
        with self.assertRaises(ValueError):
            QualityReporter(sample_fps=0)
        with self.assertRaises(ValueError):
            QualityReporter(sample_fps=-10)
