"""Unit tests for HUD layout profiles and profile registry."""

from unittest import TestCase

from pydantic import ValidationError

from gamesight.domain.models import HudLayoutProfile, HudRegion
from gamesight.perception.hud_profiles import (
    CS2_STANDARD_16X9,
    HudProfileRegistry,
    default_registry,
)


class HudRegionTests(TestCase):
    """Unit tests for the HudRegion domain model."""

    def test_valid_region_creation(self) -> None:
        region = HudRegion(
            name="minimap",
            anchor="top_left",
            x_norm=0.01,
            y_norm=0.01,
            w_norm=0.18,
            h_norm=0.30,
            description="Radar",
        )
        self.assertEqual(region.name, "minimap")
        self.assertEqual(region.anchor, "top_left")
        self.assertEqual(region.description, "Radar")

    def test_to_pixel_1920x1080(self) -> None:
        region = HudRegion(
            name="minimap", anchor="top_left",
            x_norm=0.01, y_norm=0.01, w_norm=0.18, h_norm=0.30,
        )
        x, y, w, h = region.to_pixel(1920, 1080)
        self.assertEqual(x, 19)   # 0.01 * 1920
        self.assertEqual(y, 10)   # 0.01 * 1080
        self.assertEqual(w, 345)  # 0.18 * 1920
        self.assertEqual(h, 324)  # 0.30 * 1080

    def test_to_pixel_1280x720(self) -> None:
        region = HudRegion(
            name="crosshair", anchor="center",
            x_norm=0.48, y_norm=0.47, w_norm=0.04, h_norm=0.06,
        )
        x, y, w, h = region.to_pixel(1280, 720)
        self.assertEqual(x, 614)  # 0.48 * 1280
        self.assertEqual(y, 338)  # 0.47 * 720
        self.assertEqual(w, 51)   # 0.04 * 1280
        self.assertEqual(h, 43)   # 0.06 * 720

    def test_to_pixel_2560x1440(self) -> None:
        region = HudRegion(
            name="weapon", anchor="bottom_right",
            x_norm=0.78, y_norm=0.88, w_norm=0.21, h_norm=0.11,
        )
        x, y, w, h = region.to_pixel(2560, 1440)
        self.assertEqual(x, 1996)
        self.assertEqual(y, 1267)
        self.assertEqual(w, 537)
        self.assertEqual(h, 158)

    def test_to_pixel_minimum_dimensions_never_zero(self) -> None:
        """Even tiny normalized values produce at least 1 px width/height."""
        region = HudRegion(
            name="tiny", anchor="center",
            x_norm=0.0, y_norm=0.0, w_norm=0.00001, h_norm=0.00001,
        )
        x, y, w, h = region.to_pixel(1920, 1080)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertGreaterEqual(w, 1)
        self.assertGreaterEqual(h, 1)

    def test_x_norm_out_of_range_raises(self) -> None:
        with self.assertRaises(ValidationError):
            HudRegion(
                name="bad", anchor="top_left",
                x_norm=1.5, y_norm=0.5, w_norm=0.1, h_norm=0.1,
            )

    def test_y_norm_out_of_range_raises(self) -> None:
        with self.assertRaises(ValidationError):
            HudRegion(
                name="bad", anchor="top_left",
                x_norm=0.5, y_norm=-0.2, w_norm=0.1, h_norm=0.1,
            )

    def test_w_norm_out_of_range_raises(self) -> None:
        with self.assertRaises(ValidationError):
            HudRegion(
                name="bad", anchor="top_left",
                x_norm=0.5, y_norm=0.5, w_norm=1.2, h_norm=0.1,
            )

    def test_h_norm_out_of_range_raises(self) -> None:
        with self.assertRaises(ValidationError):
            HudRegion(
                name="bad", anchor="top_left",
                x_norm=0.5, y_norm=0.5, w_norm=0.1, h_norm=1.2,
            )

    def test_region_serialisation_roundtrip(self) -> None:
        region = HudRegion(
            name="kill_feed", anchor="top_right",
            x_norm=0.78, y_norm=0.01, w_norm=0.21, h_norm=0.25,
            description="Kill feed",
        )
        data = region.model_dump()
        reloaded = HudRegion(**data)
        self.assertEqual(reloaded.name, region.name)
        self.assertEqual(reloaded.anchor, region.anchor)
        self.assertEqual(reloaded.to_pixel(1920, 1080), region.to_pixel(1920, 1080))

    def test_description_defaults_to_empty_string(self) -> None:
        region = HudRegion(
            name="test", anchor="center",
            x_norm=0.5, y_norm=0.5, w_norm=0.1, h_norm=0.1,
        )
        self.assertEqual(region.description, "")


class HudLayoutProfileTests(TestCase):
    """Unit tests for HudLayoutProfile."""

    def test_empty_profile(self) -> None:
        profile = HudLayoutProfile(name="empty", game="cs2", aspect_ratio="16:9")
        self.assertEqual(profile.region_names, [])
        self.assertIsNone(profile.region("anything"))

    def test_region_lookup_by_name(self) -> None:
        r1 = HudRegion(
            name="a", anchor="top_left",
            x_norm=0.1, y_norm=0.1, w_norm=0.1, h_norm=0.1,
        )
        r2 = HudRegion(
            name="b", anchor="top_right",
            x_norm=0.8, y_norm=0.1, w_norm=0.1, h_norm=0.1,
        )
        profile = HudLayoutProfile(
            name="test", game="cs2", aspect_ratio="16:9", regions=[r1, r2],
        )
        self.assertIs(profile.region("a"), r1)
        self.assertIs(profile.region("b"), r2)
        self.assertIsNone(profile.region("c"))

    def test_region_names_preserves_order(self) -> None:
        regions = [
            HudRegion(name=n, anchor="center", x_norm=0.5, y_norm=0.5, w_norm=0.1, h_norm=0.1)
            for n in ("alpha", "beta", "gamma")
        ]
        profile = HudLayoutProfile(
            name="ordered", game="cs2", aspect_ratio="16:9", regions=regions,
        )
        self.assertEqual(profile.region_names, ["alpha", "beta", "gamma"])

    def test_serialisation_roundtrip(self) -> None:
        profile = CS2_STANDARD_16X9
        data = profile.model_dump()
        reloaded = HudLayoutProfile(**data)
        self.assertEqual(reloaded.name, profile.name)
        self.assertEqual(reloaded.game, profile.game)
        self.assertEqual(reloaded.aspect_ratio, profile.aspect_ratio)
        self.assertEqual(len(reloaded.regions), len(profile.regions))
        for orig, restored in zip(profile.regions, reloaded.regions):
            self.assertEqual(orig.name, restored.name)
            self.assertEqual(orig.anchor, restored.anchor)


class CS2Standard16x9ProfileTests(TestCase):
    """Validation tests for the built-in CS2 16:9 profile."""

    def test_profile_has_expected_name(self) -> None:
        self.assertEqual(CS2_STANDARD_16X9.name, "cs2_standard_16x9")

    def test_profile_game_is_cs2(self) -> None:
        self.assertEqual(CS2_STANDARD_16X9.game, "cs2")

    def test_profile_aspect_ratio_is_16_9(self) -> None:
        self.assertEqual(CS2_STANDARD_16X9.aspect_ratio, "16:9")

    def test_profile_contains_seven_regions(self) -> None:
        self.assertEqual(len(CS2_STANDARD_16X9.regions), 7)

    def test_expected_region_names_present(self) -> None:
        expected = {
            "minimap", "round_info", "kill_feed", "crosshair",
            "money", "player_status", "weapon_utility",
        }
        self.assertEqual(set(CS2_STANDARD_16X9.region_names), expected)

    def test_all_regions_have_valid_anchors(self) -> None:
        valid_anchors = {
            "top_left", "top_center", "top_right",
            "center", "bottom_left", "bottom_center", "bottom_right",
        }
        for region in CS2_STANDARD_16X9.regions:
            with self.subTest(region_name=region.name):
                self.assertIn(region.anchor, valid_anchors)

    def test_all_regions_have_coordinates_in_normalized_range(self) -> None:
        for region in CS2_STANDARD_16X9.regions:
            for attr, label in [
                ("x_norm", "x"), ("y_norm", "y"),
                ("w_norm", "width"), ("h_norm", "height"),
            ]:
                with self.subTest(region_name=region.name, attr=label):
                    value = getattr(region, attr)
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_regions_do_not_overlap_their_quadrants(self) -> None:
        """Regions with different anchors must not overlap unreasonably.

        Checks that regions anchored in different quadrants are not
        occupying the same screen space.  This is a sanity check, not
        a precise pixel-level assertion.
        """
        minimap = CS2_STANDARD_16X9.region("minimap")
        kill_feed = CS2_STANDARD_16X9.region("kill_feed")
        self.assertLess(
            minimap.x_norm + minimap.w_norm,
            kill_feed.x_norm,
            "minimap should be fully left of kill_feed",
        )

        money = CS2_STANDARD_16X9.region("money")
        weapon = CS2_STANDARD_16X9.region("weapon_utility")
        self.assertLess(
            money.x_norm + money.w_norm,
            weapon.x_norm,
            "money should be fully left of weapon_utility",
        )

    def test_crosshair_is_roughly_centered(self) -> None:
        crosshair = CS2_STANDARD_16X9.region("crosshair")
        center_x = crosshair.x_norm + crosshair.w_norm / 2
        center_y = crosshair.y_norm + crosshair.h_norm / 2
        self.assertAlmostEqual(center_x, 0.5, delta=0.05)
        self.assertAlmostEqual(center_y, 0.5, delta=0.05)


class HudProfileRegistryTests(TestCase):
    """Unit tests for HudProfileRegistry."""

    def setUp(self) -> None:
        self.registry = HudProfileRegistry()

    def test_empty_registry(self) -> None:
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.registry.list_names(), [])
        self.assertIsNone(self.registry.get("anything"))

    def test_register_and_retrieve(self) -> None:
        profile = HudLayoutProfile(name="p1", game="cs2", aspect_ratio="16:9")
        self.registry.register(profile)
        self.assertIs(self.registry.get("p1"), profile)

    def test_register_overwrites_existing(self) -> None:
        first = HudLayoutProfile(name="p1", game="cs2", aspect_ratio="16:9")
        second = HudLayoutProfile(name="p1", game="cs2", aspect_ratio="21:9")
        self.registry.register(first)
        self.registry.register(second)
        self.assertIs(self.registry.get("p1"), second)

    def test_list_names_returns_insertion_order(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            self.registry.register(
                HudLayoutProfile(name=name, game="cs2", aspect_ratio="16:9")
            )
        self.assertEqual(self.registry.list_names(), ["alpha", "beta", "gamma"])

    def test_contains(self) -> None:
        self.registry.register(
            HudLayoutProfile(name="cs2", game="cs2", aspect_ratio="16:9")
        )
        self.assertIn("cs2", self.registry)
        self.assertNotIn("valorant", self.registry)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_default_registry_comes_preloaded(self) -> None:
        registry = default_registry()
        self.assertIn("cs2_standard_16x9", registry)
        self.assertEqual(len(registry), 1)
        profile = registry.get("cs2_standard_16x9")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.game, "cs2")
