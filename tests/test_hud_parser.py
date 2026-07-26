"""Unit tests for CS2HudParser."""

from unittest import TestCase

import numpy as np

from gamesight.domain.models import HudLayoutProfile, HudRegion
from gamesight.perception.extractors import (
    CrosshairExtractor,
    HPBarExtractor,
    KillFeedExtractor,
    MoneyExtractor,
    RoundInfoExtractor,
)
from gamesight.perception.hud_parser import CS2HudParser, HudParser
from gamesight.perception.hud_profiles import CS2_STANDARD_16X9


class CS2HudParserInterfaceTests(TestCase):
    """Basic contract and wiring tests."""

    def test_implements_hud_parser(self) -> None:
        parser = CS2HudParser(CS2_STANDARD_16X9)
        self.assertIsInstance(parser, HudParser)

    def test_identify_profile_returns_profile_name(self) -> None:
        parser = CS2HudParser(CS2_STANDARD_16X9)
        self.assertEqual(parser.identify_profile(None), "cs2_standard_16x9")

    def test_parse_non_numpy_frame_returns_error_state(self) -> None:
        parser = CS2HudParser(CS2_STANDARD_16X9)
        state = parser.parse("not an array", 0, 0.0)
        self.assertEqual(state.profile, "cs2_standard_16x9")
        self.assertIn("_error", state.values)
        self.assertEqual(state.confidence, 0.0)

    def test_parse_no_extractors_returns_empty_values(self) -> None:
        parser = CS2HudParser(CS2_STANDARD_16X9)  # no extractors registered
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        state = parser.parse(frame, 5, 1.5)
        self.assertEqual(state.frame_index, 5)
        self.assertAlmostEqual(state.timestamp_sec, 1.5)
        self.assertEqual(state.values, {})
        self.assertEqual(state.confidence, 0.0)


class CS2HudParserIntegrationTests(TestCase):
    """End-to-end tests with real extractors on synthetic frames."""

    def setUp(self) -> None:
        self.profile = CS2_STANDARD_16X9
        self.extractors = {
            "crosshair": CrosshairExtractor(),
            "player_status": HPBarExtractor(),
            "kill_feed": KillFeedExtractor(),
            "money": MoneyExtractor(),
            "round_info": RoundInfoExtractor(),
        }
        self.parser = CS2HudParser(self.profile, self.extractors)

    def _blank_frame(self) -> np.ndarray:
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_blank_frame_all_extractors_report_false(self) -> None:
        state = self.parser.parse(self._blank_frame(), 0, 0.0)
        self.assertIn("crosshair.crosshair_visible", state.values)
        self.assertFalse(state.values["crosshair.crosshair_visible"])
        self.assertIn("player_status.hp", state.values)
        self.assertEqual(state.values["player_status.hp"], 0)
        self.assertIn("kill_feed.kill_feed_active", state.values)
        self.assertFalse(state.values["kill_feed.kill_feed_active"])
        self.assertIn("money.money_visible", state.values)
        self.assertFalse(state.values["money.money_visible"])
        self.assertIn("round_info.round_active", state.values)
        self.assertFalse(state.values["round_info.round_active"])

    def test_crosshair_on_dark_background_detected(self) -> None:
        frame = self._blank_frame()
        # Draw crosshair lines at centre
        cx, cy = 960, 540
        frame[cy - 2 : cy + 2, cx - 20 : cx + 20, :] = [50, 220, 50]
        frame[cy - 15 : cy + 15, cx - 2 : cx + 2, :] = [50, 220, 50]
        state = self.parser.parse(frame, 0, 0.0)
        self.assertTrue(state.values["crosshair.crosshair_visible"])

    def test_hp_bar_full_green_detected(self) -> None:
        frame = self._blank_frame()
        # player_status region at 1920x1080: x=556, y=955, w=806, h=118
        # HP bar zone is bottom 45 %: rows 955+65 .. 955+118
        green = (50, 220, 50)
        frame[1020:1073, 556:1362, :] = green
        state = self.parser.parse(frame, 0, 0.0)
        self.assertGreaterEqual(state.values["player_status.hp"], 45)

    def test_armour_blue_detected(self) -> None:
        frame = self._blank_frame()
        blue = (160, 100, 20)
        # New player_status at 1920: x=547,y=855,w=826,h=225. Armour top 25%: rows 855..911
        frame[870:900, 600:1300, :] = blue
        state = self.parser.parse(frame, 0, 0.0)
        self.assertTrue(state.values["player_status.armour"])

    def test_kill_feed_with_text_detected(self) -> None:
        frame = self._blank_frame()
        # kill_feed region at 1920x1080: x=1488, y=5, w=412, h=259
        white = (255, 255, 255)
        frame[20:30, 1500:1560, :] = white
        frame[50:60, 1500:1580, :] = white
        state = self.parser.parse(frame, 0, 0.0)
        self.assertTrue(state.values["kill_feed.kill_feed_active"])

    def test_money_text_detected(self) -> None:
        frame = self._blank_frame()
        # money region at 1920x1080: x=9, y=1015, w=230, h=59
        yellow = (20, 220, 220)
        frame[1025:1040, 30:100, :] = yellow
        state = self.parser.parse(frame, 0, 0.0)
        self.assertTrue(state.values["money.money_visible"])

    def test_round_info_active_detected(self) -> None:
        frame = self._blank_frame()
        white = (255, 255, 255)
        # New round_info at 1920: x=908,y=3,w=104,h=84. Timer zone top 60%: y=3..53
        frame[15:40, 930:990, :] = white
        state = self.parser.parse(frame, 0, 0.0)
        self.assertTrue(state.values["round_info.round_active"])

    def test_values_are_namespaced_by_region(self) -> None:
        frame = self._blank_frame()
        # Draw crosshair
        cx, cy = 960, 540
        frame[cy - 2 : cy + 2, cx - 20 : cx + 20, :] = [50, 220, 50]
        frame[cy - 15 : cy + 15, cx - 2 : cx + 2, :] = [50, 220, 50]
        state = self.parser.parse(frame, 0, 0.0)

        # Every key should be prefixed with a region name.
        for key in state.values:
            self.assertIn(".", key, f"Key '{key}' should be namespaced")


class CS2HudParserCustomProfileTests(TestCase):
    """Tests with a minimal custom profile to verify profile-agnostic behaviour."""

    def test_custom_profile_only_extracts_registered_regions(self) -> None:
        profile = HudLayoutProfile(
            name="mini",
            game="test",
            aspect_ratio="16:9",
            regions=[
                HudRegion(
                    name="crosshair", anchor="center",
                    x_norm=0.48, y_norm=0.47, w_norm=0.04, h_norm=0.06,
                ),
                HudRegion(
                    name="unused", anchor="top_left",
                    x_norm=0.0, y_norm=0.0, w_norm=0.1, h_norm=0.1,
                ),
            ],
        )
        extractors = {"crosshair": CrosshairExtractor()}
        parser = CS2HudParser(profile, extractors)

        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Draw crosshair
        frame[280:320, 380:420, :] = [50, 220, 50]

        state = parser.parse(frame, 0, 0.0)
        self.assertIn("crosshair.crosshair_visible", state.values)
        # 'unused' has no extractor — should not appear
        unused_keys = [k for k in state.values if k.startswith("unused.")]
        self.assertEqual(unused_keys, [])
