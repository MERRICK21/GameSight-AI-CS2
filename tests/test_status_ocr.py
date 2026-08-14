"""Tests for conservative single-frame status-number parsing."""

from unittest import TestCase

from gamesight.perception.ocr import _parse_armour_value


class ArmourValueParsingTests(TestCase):
    def test_shield_prefix_is_removed_from_full_armour(self):
        self.assertEqual(_parse_armour_value("6100", 0.5), 100)
        self.assertEqual(_parse_armour_value("61004", 0.1), 100)

    def test_shield_prefix_is_removed_from_fifty_armour(self):
        self.assertEqual(_parse_armour_value("650", 0.5), 50)

    def test_unreliable_noise_is_unknown(self):
        self.assertIsNone(_parse_armour_value("6", 0.2))
