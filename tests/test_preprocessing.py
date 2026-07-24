"""Unit tests for video quality assessment and normalization."""

from pathlib import Path
from unittest import TestCase

from gamesight.domain.models import VideoInput, VideoMetadata
from gamesight.preprocessing.normalizer import (
    QualityDiagnostic,
    VideoPreprocessor,
)


class QualityDiagnosticTests(TestCase):
    """Tests for the QualityDiagnostic dataclass serialisation."""

    def test_to_dict_includes_all_fields(self) -> None:
        diag = QualityDiagnostic(
            width=1920,
            height=1080,
            fps=60.0,
            target_width=1920,
            target_height=1080,
            target_fps=60.0,
            resolution_match=True,
            aspect_ratio_match=True,
            fps_match=True,
            letterbox_needed=False,
            pillarbox_needed=False,
            warnings=["test warning"],
        )
        d = diag.to_dict()

        self.assertEqual(d["width"], 1920)
        self.assertEqual(d["height"], 1080)
        self.assertEqual(d["fps"], 60.0)
        self.assertTrue(d["resolution_match"])
        self.assertTrue(d["aspect_ratio_match"])
        self.assertTrue(d["fps_match"])
        self.assertFalse(d["letterbox_needed"])
        self.assertFalse(d["pillarbox_needed"])
        self.assertEqual(d["warnings"], ["test warning"])


class VideoPreprocessorTests(TestCase):
    """Tests for VideoPreprocessor.assess_quality() and normalize()."""

    def _make_video(self, video_id: str = "v1") -> VideoInput:
        return VideoInput(video_id=video_id, path=Path(f"{video_id}.mp4"))

    def _make_metadata(
        self,
        *,
        width: int | None = 1920,
        height: int | None = 1080,
        fps: float | None = 60.0,
    ) -> VideoMetadata:
        return VideoMetadata(width=width, height=height, fps=fps)

    # -- matching video -------------------------------------------------

    def test_all_checks_pass_for_perfect_input(self) -> None:
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(),
        )

        self.assertTrue(result["resolution_match"])
        self.assertTrue(result["aspect_ratio_match"])
        self.assertTrue(result["fps_match"])
        self.assertFalse(result["letterbox_needed"])
        self.assertFalse(result["pillarbox_needed"])
        self.assertEqual(result["warnings"], [])

    # -- resolution mismatch --------------------------------------------

    def test_flags_resolution_mismatch(self) -> None:
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=1280, height=720),
        )

        self.assertFalse(result["resolution_match"])
        self.assertTrue(len(result["warnings"]) >= 1)
        self.assertIn("Resolution 1280x720", result["warnings"][0])

    # -- aspect ratio mismatch ------------------------------------------

    def test_flags_aspect_ratio_mismatch_and_letterbox(self) -> None:
        """2560x1080 (ultrawide) is wider than 16:9 -> letterbox."""
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=2560, height=1080),
        )

        self.assertFalse(result["aspect_ratio_match"])
        self.assertTrue(result["letterbox_needed"])
        self.assertFalse(result["pillarbox_needed"])

    def test_flags_pillarbox_for_narrower_aspect(self) -> None:
        """1440x1080 is 4:3, narrower than 16:9."""
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=1440, height=1080),
        )

        self.assertFalse(result["aspect_ratio_match"])
        self.assertFalse(result["letterbox_needed"])
        self.assertTrue(result["pillarbox_needed"])

    # -- FPS mismatch ---------------------------------------------------

    def test_flags_fps_mismatch(self) -> None:
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(fps=30.0),
        )

        self.assertFalse(result["fps_match"])
        self.assertTrue(any("30" in w for w in result["warnings"]))

    # -- unknown metadata -----------------------------------------------

    def test_flags_unknown_resolution(self) -> None:
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=None, height=None),
        )

        self.assertFalse(result["resolution_match"])
        self.assertTrue(
            any("unknown" in w.lower() for w in result["warnings"])
        )

    def test_flags_unknown_fps(self) -> None:
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(fps=None),
        )

        self.assertFalse(result["fps_match"])
        self.assertTrue(
            any("unknown" in w.lower() for w in result["warnings"])
        )

    # -- tolerance ------------------------------------------------------

    def test_tolerance_allows_minor_rounding_errors(self) -> None:
        """1918x1078 is within the +-4 pixel tolerance."""
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=1918, height=1078),
        )

        self.assertTrue(result["resolution_match"])
        self.assertTrue(result["aspect_ratio_match"])

    def test_tolerance_rejects_large_deviations(self) -> None:
        """1915x1075 is outside the +-4 pixel tolerance."""
        preprocessor = VideoPreprocessor()
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=1915, height=1075),
        )

        self.assertFalse(result["resolution_match"])

    # -- custom targets -------------------------------------------------

    def test_accepts_custom_target_specification(self) -> None:
        preprocessor = VideoPreprocessor(
            target_width=1280, target_height=720, target_fps=30.0
        )
        result = preprocessor.assess_quality(
            self._make_video(),
            self._make_metadata(width=1280, height=720, fps=30.0),
        )

        self.assertTrue(result["resolution_match"])
        self.assertTrue(result["aspect_ratio_match"])
        self.assertTrue(result["fps_match"])

    # -- normalize ------------------------------------------------------

    def test_normalize_returns_original_input_unchanged(self) -> None:
        preprocessor = VideoPreprocessor()
        video = self._make_video()
        metadata = self._make_metadata()

        result = preprocessor.normalize(video, metadata)

        self.assertIs(result, video)
        self.assertEqual(result.video_id, "v1")
        self.assertEqual(result.path, Path("v1.mp4"))

    def test_normalize_returns_input_even_when_quality_is_poor(self) -> None:
        """normalize() should not raise or transform; it just passes through."""
        preprocessor = VideoPreprocessor()
        video = self._make_video()
        metadata = self._make_metadata(width=640, height=480, fps=15.0)

        result = preprocessor.normalize(video, metadata)

        self.assertIs(result, video)
