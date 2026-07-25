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
# CS2 HUD colours are intentionally saturated and consistent across
# resolutions, making BGR thresholding reliable for UI elements.

_HP_GREEN_LOW = np.array([30, 140, 0], dtype=np.uint8)
_HP_GREEN_HIGH = np.array([120, 255, 120], dtype=np.uint8)

_HP_RED_LOW = np.array([0, 0, 180], dtype=np.uint8)
_HP_RED_HIGH = np.array([90, 90, 255], dtype=np.uint8)

_ARMOUR_BLUE_LOW = np.array([130, 60, 0], dtype=np.uint8)
_ARMOUR_BLUE_HIGH = np.array([255, 160, 80], dtype=np.uint8)

_CROSSHAIR_GREEN_LOW = np.array([30, 150, 0], dtype=np.uint8)
_CROSSHAIR_GREEN_HIGH = np.array([120, 255, 120], dtype=np.uint8)

_WHITE_TEXT_LOW = np.array([190, 190, 190], dtype=np.uint8)
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
# The crosshair sits dead-centre and is drawn as a few bright lines with
# a small gap.  We detect it by checking whether the centre region
# contains high-contrast structure (variance significantly above noise).


class CrosshairExtractor(RegionExtractor):
    """Detect crosshair presence via central-region intensity variance.

    The crosshair introduces sharp transitions (bright lines on varied
    background), so a higher standard deviation in the centre crop is a
    strong signal.
    """

    _VARIANCE_THRESHOLD = 25.0

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
# The HP bar fills left-to-right with saturated green; it turns red when
# HP drops below ~25.  Armour is a thinner blue bar above the HP bar.
# We count green (or red) pixels in the bottom portion of the region and
# estimate HP as the ratio of coloured pixels to the bar's expected width.


class HPBarExtractor(RegionExtractor):
    """Estimate HP and armour presence from the bottom-centre HUD bar.

    The extractor scans the lower half of the region for green (HP) or
    red (low HP) pixels and the upper portion for blue (armour) pixels.
    """

    _HP_BAR_HEIGHT_FRAC = 0.45  # bottom 45 % of region contains the HP bar
    _ARMOUR_FRAC = 0.25         # armour bar sits roughly 25 % from top of region

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        h, w = region_image.shape[:2]
        if h == 0 or w == 0:
            return {"hp": 0, "hp_low": False, "armour": False}

        # HP bar — bottom portion
        hp_slice = region_image[int(h * (1 - self._HP_BAR_HEIGHT_FRAC)):, :, :]
        green_mask = cv_in_range(hp_slice, _HP_GREEN_LOW, _HP_GREEN_HIGH)
        red_mask = cv_in_range(hp_slice, _HP_RED_LOW, _HP_RED_HIGH)

        green_px = int(np.sum(green_mask > 0))
        red_px = int(np.sum(red_mask > 0))

        # The bar width is roughly the region width; estimate HP from fill ratio.
        bar_height = hp_slice.shape[0]
        expected_max_pixels = w * bar_height
        hp_ratio = (green_px + red_px) / max(expected_max_pixels, 1)
        hp = min(100, max(0, int(round(hp_ratio * 100))))
        hp_low = red_px > green_px

        # Armour bar — upper portion
        armour_slice = region_image[: int(h * self._ARMOUR_FRAC), :, :]
        blue_mask = cv_in_range(armour_slice, _ARMOUR_BLUE_LOW, _ARMOUR_BLUE_HIGH)
        armour = int(np.sum(blue_mask > 0)) > _TEXT_PIXEL_THRESHOLD

        return {"hp": hp, "hp_low": hp_low, "armour": armour}


# -- kill feed ----------------------------------------------------------------
# The kill feed at top-right is a stack of bright (white/yellow) text
# lines.  We detect activity by counting bright pixels.


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
        bright_pixels = int(np.sum(white_mask > 0)) + int(np.sum(yellow_mask > 0))

        total = max(region_image.size // 3, 1)
        fraction = bright_pixels / total

        return {
            "kill_feed_active": fraction > _ACTIVITY_THRESHOLD,
            "bright_pixel_fraction": round(fraction, 5),
        }


# -- money --------------------------------------------------------------------
# Money text at bottom-left is rendered in a light green / yellow tone.
# OCR will eventually read the actual value; for now we only report
# whether the text is present.


class MoneyExtractor(RegionExtractor):
    """Detect whether the money text is visible (OCR-ready in a future sprint)."""

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
# The round-info bar (timer + scores) at top-centre contains bright text
# when a round is active.  Between rounds the region is mostly dark.


class RoundInfoExtractor(RegionExtractor):
    """Extract round info: timer + score colours.

    Top half = timer (white), bottom half = scores (CT blue, T yellow).
    """

    def extract(
        self,
        region_image: NDArray[np.uint8],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, object]:
        if region_image.size == 0:
            return {"round_active": False, "scores_visible": False,
                    "ct_score_present": False, "t_score_present": False}

        h = region_image.shape[0]
        timer_zone = region_image[: h // 2, :]
        white_mask = cv_in_range(timer_zone, _WHITE_TEXT_LOW, _WHITE_TEXT_HIGH)
        timer_active = int(np.sum(white_mask > 0)) > _TEXT_PIXEL_THRESHOLD

        score_zone = region_image[h // 2:, :]
        ct_present = int(np.sum(cv_in_range(score_zone, _CT_SCORE_LOW, _CT_SCORE_HIGH) > 0)) > _SCORE_PIXEL_MIN
        t_present = int(np.sum(cv_in_range(score_zone, _T_SCORE_LOW, _T_SCORE_HIGH) > 0)) > _SCORE_PIXEL_MIN

        return {
            "round_active": timer_active,
            "timer_visible": timer_active,
            "ct_score_present": ct_present,
            "t_score_present": t_present,
            "scores_visible": ct_present and t_present,
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
