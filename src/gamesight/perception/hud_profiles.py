"""Built-in HUD layout profiles and a lightweight profile registry."""

from __future__ import annotations

from gamesight.domain.models import HudLayoutProfile, HudRegion


# -- CS2 standard 16:9 profile ------------------------------------------------
# Coordinates are fractions of screen dimensions, calibrated against the
# standard CS2 HUD layout.  Anchor points encode logical positioning so
# the Event Engine can reason about HUD structure without pixel math.
#
# Regions and their approximate coverage:
#   minimap        top-left     radar / mini-map
#   round_info     top-center   timer, team scores, player avatars
#   kill_feed      top-right    stacked kill / death / assist messages
#   crosshair      center       player crosshair
#   money          bottom-left  current money display
#   player_status  bottom-cent  HP bar, armour, kill counter, ammo
#   weapon_utility bottom-right weapon name, utility / grenade icons

CS2_16X9_REGIONS: list[HudRegion] = [
    HudRegion(
        name="minimap",
        anchor="top_left",
        x_norm=0.010,
        y_norm=0.010,
        w_norm=0.175,
        h_norm=0.295,
        description="Radar mini-map in the top-left corner",
    ),
    HudRegion(
        name="round_info",
        anchor="top_center",
        x_norm=0.340,
        y_norm=0.005,
        w_norm=0.320,
        h_norm=0.085,
        description="Round timer, team scores, and player avatars at top-centre",
    ),
    HudRegion(
        name="kill_feed",
        anchor="top_right",
        x_norm=0.775,
        y_norm=0.005,
        w_norm=0.215,
        h_norm=0.240,
        description="Kill / death / assist event feed at top-right",
    ),
    HudRegion(
        name="crosshair",
        anchor="center",
        x_norm=0.480,
        y_norm=0.470,
        w_norm=0.040,
        h_norm=0.060,
        description="Player crosshair at screen centre",
    ),
    HudRegion(
        name="money",
        anchor="bottom_left",
        x_norm=0.005,
        y_norm=0.940,
        w_norm=0.120,
        h_norm=0.055,
        description="Current money counter at bottom-left",
    ),
    HudRegion(
        name="player_status",
        anchor="bottom_center",
        x_norm=0.290,
        y_norm=0.885,
        w_norm=0.420,
        h_norm=0.110,
        description="HP, armour, kill count, and ammo at bottom-centre",
    ),
    HudRegion(
        name="weapon_utility",
        anchor="bottom_right",
        x_norm=0.775,
        y_norm=0.885,
        w_norm=0.215,
        h_norm=0.110,
        description="Weapon name, ammo reserve, and utility / grenade icons at bottom-right",
    ),
]

CS2_STANDARD_16X9 = HudLayoutProfile(
    name="cs2_standard_16x9",
    game="cs2",
    aspect_ratio="16:9",
    regions=CS2_16X9_REGIONS,
)


# -- profile registry ---------------------------------------------------------

class HudProfileRegistry:
    """A simple name-based registry for ``HudLayoutProfile`` instances.

    Built-in profiles are pre-registered at import time.  Callers may
    register additional profiles (e.g. from YAML configs) at runtime.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, HudLayoutProfile] = {}

    def register(self, profile: HudLayoutProfile) -> None:
        """Register a profile, overwriting any existing entry with the same name."""
        self._profiles[profile.name] = profile

    def get(self, name: str) -> HudLayoutProfile | None:
        """Look up a profile by name; returns ``None`` when not found."""
        return self._profiles.get(name)

    def list_names(self) -> list[str]:
        """Return registered profile names in insertion order."""
        return list(self._profiles.keys())

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, name: str) -> bool:
        return name in self._profiles


def default_registry() -> HudProfileRegistry:
    """Return a registry pre-loaded with the CS2 16:9 profile."""
    registry = HudProfileRegistry()
    registry.register(CS2_STANDARD_16X9)
    return registry
