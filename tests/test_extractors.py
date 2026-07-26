"""Unit tests for per-region HUD extractors."""

from unittest import TestCase

import numpy as np

from gamesight.perception.extractors import (
    CrosshairExtractor,
    HPBarExtractor,
    KillFeedExtractor,
    MoneyExtractor,
    RegionExtractor,
    RoundInfoExtractor,
)


def _bgr_image(width: int, height: int, colour: tuple[int, int, int]) -> np.ndarray:
    """Create a solid-colour BGR uint8 image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = colour
    return img


# -- CrosshairExtractor ------------------------------------------------------

class CrosshairExtractorTests(TestCase):
    def setUp(self) -> None:
        self.extractor = CrosshairExtractor()

    def test_implements_region_extractor(self) -> None:
        self.assertIsInstance(self.extractor, RegionExtractor)

    def test_blank_region_not_visible(self) -> None:
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["crosshair_visible"])
        self.assertEqual(result["variance"], 0.0)

    def test_high_contrast_crosshair_visible(self) -> None:
        """A crosshair-like pattern (bright lines on dark background) triggers visibility."""
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        img[28:32, :, :] = [50, 220, 50]
        img[:, 38:42, :] = [50, 220, 50]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["crosshair_visible"])
        self.assertGreater(result["variance"], 30.0)

    def test_dim_uniform_region_not_visible(self) -> None:
        img = _bgr_image(80, 60, (40, 40, 40))
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["crosshair_visible"])

    def test_bright_uniform_region_not_visible(self) -> None:
        """Uniform bright region has low variance — not a crosshair."""
        img = _bgr_image(80, 60, (200, 200, 200))
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["crosshair_visible"])

    def test_empty_image_returns_false(self) -> None:
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["crosshair_visible"])
        self.assertEqual(result["variance"], 0.0)

    def test_greyscale_input_works(self) -> None:
        img = np.zeros((60, 80), dtype=np.uint8)
        img[28:32, :] = 200
        result = self.extractor.extract(img, 0, 0.0)
        self.assertIsInstance(result["crosshair_visible"], bool)


# -- HPBarExtractor ----------------------------------------------------------

class HPBarExtractorTests(TestCase):
    def setUp(self) -> None:
        self.extractor = HPBarExtractor()

    def test_implements_region_extractor(self) -> None:
        self.assertIsInstance(self.extractor, RegionExtractor)

    def test_empty_image_returns_zeros(self) -> None:
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.extractor.extract(img, 0, 0.0)
        self.assertEqual(result["hp"], 0)
        self.assertFalse(result["hp_low"])
        self.assertFalse(result["armour"])

    def test_full_green_bar_is_full_hp(self) -> None:
        """A solid green region in the HP bar zone maps to ~100 HP."""
        img = np.zeros((110, 420, 3), dtype=np.uint8)
        green = (50, 220, 50)
        img[38:61, :210, :] = green
        result = self.extractor.extract(img, 0, 0.0)
        self.assertGreaterEqual(result["hp"], 80)
        self.assertFalse(result["hp_low"])
        self.assertFalse(result["armour"])

    def test_half_green_bar_is_partial_hp(self) -> None:
        img = np.zeros((110, 420, 3), dtype=np.uint8)
        green = (50, 220, 50)
        img[38:61, :105, :] = green
        result = self.extractor.extract(img, 0, 0.0)
        self.assertGreater(result["hp"], 20)
        self.assertLess(result["hp"], 70)
        self.assertFalse(result["hp_low"])

    def test_red_bar_signals_low_hp(self) -> None:
        img = np.zeros((110, 420, 3), dtype=np.uint8)
        red = (30, 30, 230)
        img[38:61, :210, :] = red
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["hp_low"], "Red-dominated HP zone should set hp_low")

    def test_blue_in_armour_zone_signals_armour(self) -> None:
        img = np.zeros((110, 420, 3), dtype=np.uint8)
        blue = (160, 100, 20)
        img[:28, :, :] = blue
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["armour"])

    def test_no_blue_means_no_armour(self) -> None:
        img = np.zeros((110, 420, 3), dtype=np.uint8)
        green = (50, 220, 50)
        img[61:, :, :] = green
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["armour"])

    def test_hp_clamped_to_100(self) -> None:
        img = np.zeros((110, 420, 3), dtype=np.uint8)
        green = (50, 220, 50)
        img[:, :, :] = green
        result = self.extractor.extract(img, 0, 0.0)
        self.assertLessEqual(result["hp"], 100)


# -- KillFeedExtractor -------------------------------------------------------

class KillFeedExtractorTests(TestCase):
    def setUp(self) -> None:
        self.extractor = KillFeedExtractor()

    def test_implements_region_extractor(self) -> None:
        self.assertIsInstance(self.extractor, RegionExtractor)

    def test_blank_region_not_active(self) -> None:
        img = np.zeros((240, 215, 3), dtype=np.uint8)
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["kill_feed_active"])
        self.assertEqual(result["bright_pixel_fraction"], 0.0)

    def test_white_text_pixels_trigger_active(self) -> None:
        img = np.zeros((240, 215, 3), dtype=np.uint8)
        # Draw enough white pixels to exceed 0.5 % activity threshold
        img[10:20, 50:100, :] = [255, 255, 255]
        img[30:40, 50:120, :] = [255, 255, 255]
        img[50:60, 50:140, :] = [255, 255, 255]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["kill_feed_active"])

    def test_yellow_text_pixels_trigger_active(self) -> None:
        img = np.zeros((240, 215, 3), dtype=np.uint8)
        img[10:20, 50:100, :] = [20, 220, 220]
        img[30:40, 50:120, :] = [20, 220, 220]
        img[50:60, 50:140, :] = [20, 220, 220]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["kill_feed_active"])

    def test_few_bright_pixels_below_threshold(self) -> None:
        img = np.zeros((240, 215, 3), dtype=np.uint8)
        img[0, 0, :] = [255, 255, 255]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["kill_feed_active"])


# -- MoneyExtractor ----------------------------------------------------------

class MoneyExtractorTests(TestCase):
    def setUp(self) -> None:
        self.extractor = MoneyExtractor()

    def test_implements_region_extractor(self) -> None:
        self.assertIsInstance(self.extractor, RegionExtractor)

    def test_blank_region_not_visible(self) -> None:
        img = np.zeros((55, 120, 3), dtype=np.uint8)
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["money_visible"])

    def test_yellow_text_is_visible(self) -> None:
        img = np.zeros((55, 120, 3), dtype=np.uint8)
        img[10:20, 20:80, :] = [20, 220, 220]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["money_visible"])

    def test_white_text_is_visible(self) -> None:
        img = np.zeros((55, 120, 3), dtype=np.uint8)
        img[10:20, 20:80, :] = [255, 255, 255]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["money_visible"])

    def test_single_bright_pixel_not_enough(self) -> None:
        img = np.zeros((55, 120, 3), dtype=np.uint8)
        img[0, 0, :] = [255, 255, 255]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["money_visible"])


# -- RoundInfoExtractor ------------------------------------------------------

class RoundInfoExtractorTests(TestCase):
    def setUp(self) -> None:
        self.extractor = RoundInfoExtractor()

    def test_implements_region_extractor(self) -> None:
        self.assertIsInstance(self.extractor, RegionExtractor)

    def test_blank_region_not_active(self) -> None:
        img = np.zeros((85, 320, 3), dtype=np.uint8)
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["round_active"])
        self.assertFalse(result["timer_visible"])

    def test_white_text_triggers_round_active(self) -> None:
        img = np.zeros((85, 320, 3), dtype=np.uint8)
        img[20:35, 120:200, :] = [255, 255, 255]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertTrue(result["round_active"])
        self.assertTrue(result["timer_visible"])

    def test_dark_non_text_pixels_do_not_trigger(self) -> None:
        img = np.zeros((85, 320, 3), dtype=np.uint8)
        img[20:35, 120:200, :] = [100, 100, 100]
        result = self.extractor.extract(img, 0, 0.0)
        self.assertFalse(result["round_active"])
        self.assertFalse(result["timer_visible"])
