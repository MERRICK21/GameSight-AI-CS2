"""Per-region HUD state extractors for CS2.

Each extractor operates on a cropped region image (BGR numpy array) and
returns a flat dictionary of structured values.  Colour heuristics are
used in this sprint; OCR and template-matching can be swapped in later
by implementing new extractors behind the same ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


# -- colour constants (BGR) ---------------------------------------------------

_HP_GREEN_LOW = np.array([30, 140, 0], dtype=np.uint8)
_HP_GREEN_HIGH = np.array([120, 255, 120], dtype=np.uint8)

_HP_RED_LOW = np.array([0, 0, 180], dtype=np.uint8)
_HP_RED_HIGH = np.array([90, 90, 255], dtype=np.uint8)

_ARMOUR_BLUE_LOW = np.array([130, 60, 0], dtype=np.uint8)
_ARMOUR_BLUE_HIGH = np.array([255, 160, 80], dtype=np.uint8)

_CROSSHAIR_GREEN_LOW = np.array([30, 150, 0], dtype=np.uint8)
_CROSSHAIR_GREEN_HIGH = np.array([120, 255, 120], dtype=np.uint8)

_WHITE_TEXT_LOW = np.array([200, 200, 200], dtype=np.uint8)
_WHITE_TEXT_HIGH = np.array([255, 255, 255], dtype=np.uint8)

_YELLOW_TEXT_LOW = np.array([0, 180, 180], dtype=np.uint8)
_YELLOW_TEXT_HIGH = np.array([70, 255, 255], dtype=np.uint8)
_CT_SCORE_LOW = np.array([100, 120, 180], dtype=np.uint8)
_CT_SCORE_HIGH = np.array([200, 220, 255], dtype=np.uint8)
_T_SCORE_LOW = np.array([0, 140, 200], dtype=np.uint8)
_T_SCORE_HIGH = np.array([60, 220, 255], dtype=np.uint8)
_SCORE_PIXEL_MIN = 8

# Minimum fraction of non-black pixels in a region to consider it "active".
_ACTIVITY_THRESHOLD = 0.005

# Minimum number of bright pixels needed to declare HUD text visible.
_TEXT_PIXEL_THRESHOLD = 10

# Timer zone: fraction of bright-white pixels that indicates timer is visible.
_TIMER_ACTIVE_RATIO = 0.008


class RegionExtractor(ABC):
    """Contract for a single-region HUD value extractor."""

    @abstractmethod
    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        """Return structured values extracted from *region_image*."""


# -- crosshair ----------------------------------------------------------------

class CrosshairExtractor(RegionExtractor):
    """Detect crosshair presence via central-region intensity variance."""

    _VARIANCE_THRESHOLD = 12.0  # lowered for better sensitivity on 2K

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        grey = _to_grey(region_image)
        if grey.size == 0:
            return {"crosshair_visible": False, "variance": 0.0}

        variance = float(np.std(grey))
        return {
            "crosshair_visible": variance > self._VARIANCE_THRESHOLD,
            "variance": round(variance, 2),
        }


# -- HP bar -------------------------------------------------------------------

class HPBarExtractor(RegionExtractor):
    """Estimate HP and armour from the bottom-centre HUD bar.
    
    The HP bar is a thin horizontal strip with a white number above it.
    Ammo (yellow digits) sits on the right side of the same region.
    """

    _HP_BAND_TOP = 0.35
    _HP_BAND_BOTTOM = 0.55
    _HP_LEFT_FRAC = 0.50
    _ARMOUR_FRAC = 0.25

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        h, w = region_image.shape[:2]
        if h == 0 or w == 0:
            return {"hp": 0, "hp_low": False, "armour": False}

        # HP bar: thin band in left half of region.
        hp_left = max(1, int(w * self._HP_LEFT_FRAC))
        hp_slice = region_image[int(h * self._HP_BAND_TOP): int(h * self._HP_BAND_BOTTOM), :hp_left, :]
        green_mask = cv_in_range(hp_slice, _HP_GREEN_LOW, _HP_GREEN_HIGH)
        red_mask = cv_in_range(hp_slice, _HP_RED_LOW, _HP_RED_HIGH)

        green_px = int(np.sum(green_mask > 0))
        red_px = int(np.sum(red_mask > 0))

        bar_height = hp_slice.shape[0]
        expected_max_pixels = hp_left * bar_height
        hp_ratio = (green_px + red_px) / max(expected_max_pixels, 1)
        hp = min(100, max(0, int(round(hp_ratio * 100))))
        hp_low = red_px > green_px

        # Armour: upper portion, left side.
        armour_slice = region_image[: int(h * self._ARMOUR_FRAC), :hp_left, :]
        blue_mask = cv_in_range(armour_slice, _ARMOUR_BLUE_LOW, _ARMOUR_BLUE_HIGH)
        armour = int(np.sum(blue_mask > 0)) > _TEXT_PIXEL_THRESHOLD

        # Ammo: right half of region, look for yellow digits.
        ammo_slice = region_image[:, hp_left:, :]
        yellow_mask = cv_in_range(ammo_slice, _YELLOW_TEXT_LOW, _YELLOW_TEXT_HIGH)
        ammo_px = int(np.sum(yellow_mask > 0))
        ammo_visible = ammo_px > _TEXT_PIXEL_THRESHOLD * 2

        return {"hp": hp, "hp_low": hp_low, "armour": armour, "ammo_visible": ammo_visible, "ammo_pixels": ammo_px}


# -- kill feed ----------------------------------------------------------------

class KillFeedExtractor(RegionExtractor):
    """Detect kill-feed activity by counting bright text pixels."""

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        white_mask = cv_in_range(region_image, _WHITE_TEXT_LOW, _WHITE_TEXT_HIGH)
        yellow_mask = cv_in_range(region_image, _YELLOW_TEXT_LOW, _YELLOW_TEXT_HIGH)
        # Exclude green net_graph text (ping/fps) -- common in POV recordings.
        green_mask = cv_in_range(region_image, np.array([0, 120, 0], dtype=np.uint8), np.array([80, 255, 80], dtype=np.uint8))
        bright_pixels = int(np.sum(white_mask > 0)) + int(np.sum(yellow_mask > 0))
        green_px = int(np.sum(green_mask > 0))
        bright_pixels = max(0, bright_pixels - green_px)

        total = max(region_image.size // 3, 1)
        fraction = bright_pixels / total

        return {
            "kill_feed_active": fraction > _ACTIVITY_THRESHOLD,
            "bright_pixel_fraction": round(fraction, 5),
        }


# -- money --------------------------------------------------------------------

class MoneyExtractor(RegionExtractor):
    """Detect whether the money text is visible."""

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        yellow_mask = cv_in_range(region_image, _YELLOW_TEXT_LOW, _YELLOW_TEXT_HIGH)
        white_mask = cv_in_range(region_image, _WHITE_TEXT_LOW, _WHITE_TEXT_HIGH)
        bright_pixels = int(np.sum(yellow_mask > 0)) + int(np.sum(white_mask > 0))

        return {"money_visible": bright_pixels > _TEXT_PIXEL_THRESHOLD}


# -- round info ---------------------------------------------------------------

class RoundInfoExtractor(RegionExtractor):
    """Extract round info: timer activity + score colours.

    The round_info region at top-centre contains:
      - Top ~60%: white countdown timer digits on dark background
      - Bottom ~40%: CT score (blue) and T score (yellow)

    Timer detection uses a pixel-ratio threshold instead of an absolute
    pixel count, which is more robust against HUD variations.

    Between rounds the scoreboard replaces the timer, causing a
    significant drop in bright-white pixel density that serves as a
    reliable round-transition signal.
    """

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        if region_image.size == 0:
            return {
                "round_active": False, "timer_visible": False,
                "ct_score_present": False, "t_score_present": False,
                "scores_visible": False,
                "timer_pixel_ratio": 0.0,
                "ct_score_pixels": 0, "t_score_pixels": 0,
            }

        h = region_image.shape[0]

        # Timer zone: top ~60% of the round_info strip.
        timer_zone = region_image[: int(h * 0.60), :]
        timer_pixels = timer_zone.size // 3 if timer_zone.size > 0 else 1
        white_mask = cv_in_range(timer_zone, _WHITE_TEXT_LOW, _WHITE_TEXT_HIGH)
        white_count = int(np.sum(white_mask > 0))
        timer_ratio = white_count / max(timer_pixels, 1)
        # Timer is "active" when bright-white pixel ratio exceeds threshold.
        timer_active = timer_ratio > _TIMER_ACTIVE_RATIO

        # Score zone: bottom ~40%.
        score_zone = region_image[int(h * 0.60):, :]
        ct_present = int(np.sum(cv_in_range(score_zone, _CT_SCORE_LOW, _CT_SCORE_HIGH) > 0)) > _SCORE_PIXEL_MIN
        ct_px = int(np.sum(cv_in_range(score_zone, _CT_SCORE_LOW, _CT_SCORE_HIGH) > 0))
        t_present = int(np.sum(cv_in_range(score_zone, _T_SCORE_LOW, _T_SCORE_HIGH) > 0)) > _SCORE_PIXEL_MIN
        t_px = int(np.sum(cv_in_range(score_zone, _T_SCORE_LOW, _T_SCORE_HIGH) > 0))

        return {
            "round_active": timer_active,
            "timer_visible": timer_active,
            "timer_pixel_ratio": round(timer_ratio, 5),
            "ct_score_present": ct_present,
            "t_score_present": t_present,
            "scores_visible": ct_present and t_present,
            "ct_score_pixels": ct_px,
            "t_score_pixels": t_px,
        }


# -- helpers ------------------------------------------------------------------

def _to_grey(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a BGR image to greyscale, handling single-channel input."""
    if image.ndim == 2:
        return image
    return image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299


def cv_in_range(
    image: NDArray[np.uint8],
    lower: NDArray[np.uint8],
    upper: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """NumPy-only replacement for ``cv2.inRange`` so tests stay light.

    Uses the same inclusive-lower, exclusive-upper semantics as OpenCV.
    """
    if image.size == 0:
        return np.array([], dtype=np.uint8)
    return np.all((image >= lower) & (image <= upper), axis=-1).astype(np.uint8) * 255
