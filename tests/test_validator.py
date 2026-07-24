"""Unit tests for video validation."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

from gamesight.domain.models import VideoInput, VideoMetadata
from gamesight.preprocessing.validator import ValidationResult, VideoValidator


class ValidationResultTests(TestCase):
    """Tests for ValidationResult dataclass serialisation."""

    def test_to_dict_serialises_all_fields(self) -> None:
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["low resolution"],
        )
        d = result.to_dict()

        self.assertTrue(d["is_valid"])
        self.assertEqual(d["errors"], [])
        self.assertEqual(d["warnings"], ["low resolution"])

    def test_is_valid_false_when_errors_present(self) -> None:
        result = ValidationResult(
            is_valid=False,
            errors=["file not found"],
            warnings=[],
        )
        self.assertFalse(result.is_valid)
        self.assertFalse(result.to_dict()["is_valid"])


class VideoValidatorTests(TestCase):
    """Tests for VideoValidator.validate()."""

    def setUp(self) -> None:
        self.validator = VideoValidator()

    @staticmethod
    def _meta(
        *,
        width: int | None = 1920,
        height: int | None = 1080,
        fps: float | None = 60.0,
    ) -> VideoMetadata:
        """Build metadata with sensible defaults for all fields."""
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            duration_sec=300.0,
            codec="avc1",
        )

    def _video(self, ext: str = ".mp4") -> VideoInput:
        return VideoInput(video_id="v1", path=Path(f"video{ext}"))

    # -- valid input ----------------------------------------------------

    def test_valid_video_passes_all_checks(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = Path(f.name)

        try:
            video = VideoInput(video_id="ok", path=temp_path)
            result = self.validator.validate(video, self._meta())

            self.assertTrue(result.is_valid)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])
        finally:
            temp_path.unlink(missing_ok=True)

    # -- file not found -------------------------------------------------

    def test_rejects_missing_file(self) -> None:
        video = VideoInput(video_id="ghost", path=Path("nonexistent.mp4"))
        result = self.validator.validate(video, self._meta())

        self.assertFalse(result.is_valid)
        self.assertTrue(any("not found" in e for e in result.errors))

    # -- extension ------------------------------------------------------

    def test_rejects_unsupported_extension(self) -> None:
        video = self._video(ext=".avi")
        result = self.validator.validate(video, self._meta())

        self.assertFalse(result.is_valid)
        self.assertTrue(any("Unsupported" in e for e in result.errors))

    def test_accepts_mov_and_mkv(self) -> None:
        with NamedTemporaryFile(suffix=".mov", delete=False) as f:
            mov_path = Path(f.name)
        with NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            mkv_path = Path(f.name)

        try:
            for path in [mov_path, mkv_path]:
                video = VideoInput(video_id="ok", path=path)
                result = self.validator.validate(video, self._meta())
                self.assertTrue(result.is_valid)
        finally:
            mov_path.unlink(missing_ok=True)
            mkv_path.unlink(missing_ok=True)

    def test_accepts_custom_extensions(self) -> None:
        validator = VideoValidator(accepted_extensions={".avi", ".webm"})
        result = validator.validate(
            self._video(ext=".avi"), self._meta()
        )
        self.assertTrue(result.is_valid)

    # -- resolution -----------------------------------------------------

    def test_warns_on_low_resolution(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            p = Path(f.name)

        try:
            result = self.validator.validate(
                VideoInput(video_id="x", path=p),
                self._meta(width=320, height=240),
            )
            self.assertTrue(result.is_valid)
            self.assertTrue(any("below" in w for w in result.warnings))
        finally:
            p.unlink(missing_ok=True)

    def test_errors_on_unknown_resolution(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            p = Path(f.name)

        try:
            result = self.validator.validate(
                VideoInput(video_id="x", path=p),
                self._meta(width=None, height=None),
            )
            self.assertFalse(result.is_valid)
            self.assertTrue(
                any("unknown" in e.lower() for e in result.errors)
            )
        finally:
            p.unlink(missing_ok=True)

    # -- frame rate -----------------------------------------------------

    def test_errors_on_unknown_fps(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            p = Path(f.name)

        try:
            result = self.validator.validate(
                VideoInput(video_id="x", path=p),
                self._meta(fps=None),
            )
            self.assertFalse(result.is_valid)
            self.assertTrue(
                any("unknown" in e.lower() for e in result.errors)
            )
        finally:
            p.unlink(missing_ok=True)

    def test_warns_on_low_fps(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            p = Path(f.name)

        try:
            result = self.validator.validate(
                VideoInput(video_id="x", path=p),
                self._meta(fps=0.5),
            )
            self.assertTrue(result.is_valid)
            self.assertTrue(any("below" in w for w in result.warnings))
        finally:
            p.unlink(missing_ok=True)

    def test_warns_on_unusually_high_fps(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            p = Path(f.name)

        try:
            result = self.validator.validate(
                VideoInput(video_id="x", path=p),
                self._meta(fps=300.0),
            )
            self.assertTrue(result.is_valid)
            self.assertTrue(
                any("Unusually high" in w for w in result.warnings)
            )
        finally:
            p.unlink(missing_ok=True)

    # -- custom thresholds ----------------------------------------------

    def test_custom_min_resolution_warns(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            p = Path(f.name)

        try:
            validator = VideoValidator(min_width=1280, min_height=720)
            result = validator.validate(
                VideoInput(video_id="x", path=p),
                self._meta(width=1024, height=768),
            )
            self.assertTrue(result.is_valid)
            self.assertTrue(any("below" in w for w in result.warnings))
        finally:
            p.unlink(missing_ok=True)

    # -- multiple issues ------------------------------------------------

    def test_accumulates_multiple_errors(self) -> None:
        video = VideoInput(video_id="bad", path=Path("ghost.avi"))
        result = self.validator.validate(
            video,
            self._meta(width=None, height=None, fps=None),
        )

        self.assertFalse(result.is_valid)
        self.assertGreaterEqual(len(result.errors), 3)
