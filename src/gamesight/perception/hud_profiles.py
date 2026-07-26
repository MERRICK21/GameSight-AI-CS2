"""Built-in HUD layout profiles and a lightweight profile registry.

Calibrated for 2560x1440 (2K) with ratios that scale to any 16:9 resolution.
"""

from __future__ import annotations

from gamesight.domain.models import HudLayoutProfile, HudRegion


CS2_16X9_REGIONS: list[HudRegion] = [
    HudRegion(name="minimap", anchor="top_left",
              x_norm=0.000, y_norm=0.000, w_norm=0.257, h_norm=0.343,
              description="Radar mini-map in the top-left corner"),
    HudRegion(name="round_info", anchor="top_center",
              x_norm=0.473, y_norm=0.003, w_norm=0.054, h_norm=0.078,
              description="Round timer, team scores at top-centre"),
    HudRegion(name="kill_feed", anchor="top_right",
              x_norm=0.785, y_norm=0.003, w_norm=0.212, h_norm=0.250,
              description="Kill / death / assist event feed at top-right"),
    HudRegion(name="crosshair", anchor="center",
              x_norm=0.480, y_norm=0.470, w_norm=0.040, h_norm=0.060,
              description="Player crosshair at screen centre"),
    HudRegion(name="money", anchor="bottom_left",
              x_norm=0.000, y_norm=0.861, w_norm=0.137, h_norm=0.139,
              description="Current money counter at bottom-left"),
    HudRegion(name="player_status", anchor="bottom_center",
              x_norm=0.250, y_norm=0.792, w_norm=0.500, h_norm=0.208,
              description="HP, armour, kill count, and ammo at bottom-centre"),
    HudRegion(name="weapon_utility", anchor="bottom_right",
              x_norm=0.859, y_norm=0.573, w_norm=0.141, h_norm=0.427,
              description="Weapon name, ammo reserve, utility icons at bottom-right"),
]
CS2_STANDARD_16X9 = HudLayoutProfile(name="cs2_standard_16x9", game="cs2",
                                      aspect_ratio="16:9", regions=CS2_16X9_REGIONS)

class HudProfileRegistry:
    def __init__(self): self._profiles: dict[str, HudLayoutProfile] = {}
    def register(self, p): self._profiles[p.name] = p
    def get(self, name): return self._profiles.get(name)
    def list_names(self): return list(self._profiles.keys())
    def __len__(self): return len(self._profiles)
    def __contains__(self, name): return name in self._profiles

def default_registry():
    r = HudProfileRegistry(); r.register(CS2_STANDARD_16X9); return r
