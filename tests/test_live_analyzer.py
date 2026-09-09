"""Tests for evidence-safe single-screenshot advice."""

from unittest import TestCase

import numpy as np

from gamesight.domain.models import HudState
from gamesight.live.analyzer import LiveAnalyzer


class _Parser:
    def __init__(self, values):
        self._values = values

    def parse(self, _image, frame_index, timestamp_sec):
        return HudState(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            profile="test",
            values=self._values,
            confidence=1.0,
        )


class LiveAnalyzerArmourTests(TestCase):
    def _analyse(self, armour_value=None, armour_presence=False):
        values = {
            "player_status.hp": 100,
            "player_status.armour": armour_presence,
            "player_status.armour_value": armour_value,
            "round_info.round_active": True,
            "crosshair.crosshair_visible": True,
        }
        return LiveAnalyzer(_Parser(values)).analyze(
            np.zeros((32, 32, 3), dtype=np.uint8)
        )

    def test_boolean_armour_presence_is_not_treated_as_numeric_value(self):
        for presence in (False, True):
            advice = self._analyse(armour_value=None, armour_presence=presence)
            self.assertFalse(any("Armour is low" in tip for tip in advice.tips))

    def test_full_armour_does_not_generate_purchase_advice(self):
        advice = self._analyse(armour_value=100, armour_presence=True)
        self.assertFalse(any("Armour is low" in tip for tip in advice.tips))

    def test_fifty_armour_recommends_refill_and_helmet(self):
        advice = self._analyse(armour_value=50, armour_presence=True)
        low_tip = next(tip for tip in advice.tips if "Armour is low" in tip)
        self.assertIn("50", low_tip)
        self.assertIn("helmet", low_tip)

    def test_default_action_and_crosshair_tip_do_not_infer_motion(self):
        advice = self._analyse(armour_value=100, armour_presence=True)
        self.assertIn("one screenshot cannot determine", advice.next_action)
        crosshair_tip = next(
            tip for tip in advice.tips if "crosshair is visible" in tip.lower()
        )
        self.assertIn("cannot establish", crosshair_tip)
        self.assertNotIn("steady", crosshair_tip.lower())
