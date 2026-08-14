"""Live screenshot analysis -- single-frame CS2 tactical advice.

Analyses a single CS2 screenshot (HUD state, minimap, crosshair, HP)
and produces actionable CS2-specific tactical suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gamesight.i18n.loader import I18nLoader


@dataclass
class LiveAdvice:
    """Tactical advice from a single-frame analysis."""
    status: str
    next_action: str
    tips: list[str]
    confidence: float

    @classmethod
    def empty(cls) -> LiveAdvice:
        return cls(status="Unable to analyze", next_action="Upload a clearer screenshot", tips=[], confidence=0.0)


class LiveAnalyzer:
    """Analyse a single CS2 screenshot and produce tactical advice.

    Uses the HUD parser to extract game state from one frame,
    then applies CS2-specific heuristics to suggest the next action.
    No medkits, no healing items -- advice is CS2-realistic.
    """

    def __init__(self, hud_parser, loader: I18nLoader | None = None) -> None:
        self._parser = hud_parser
        self._t = loader or I18nLoader("en")

    def analyze(self, image: NDArray[np.uint8], timestamp_sec: float = 0.0) -> LiveAdvice:
        state = self._parser.parse(image, 0, timestamp_sec)
        values = state.values
        t = self._t

        hp = self._get_num(values, "player_status.hp", 100)
        # ``player_status.armour`` is a presence boolean, not a point value.
        # Treating it as a number turns False/True into 0/1 in Python and made
        # every screenshot look like low armour.  Only numeric OCR may drive a
        # low-armour recommendation; unknown armour stays unknown.
        armour = self._get_optional_num(values, "player_status.armour_value")
        crosshair_visible = bool(values.get("crosshair.crosshair_visible", True))
        round_active = bool(values.get("round_info.round_active", True))
        kill_feed_active = bool(values.get("kill_feed.kill_feed_active", False))
        money_visible = bool(values.get("money.money_visible", False))

        tips: list[str] = []
        status_parts: list[str] = []
        action_parts: list[str] = []

        # HP analysis (CS2: no healing -- only survival matters)
        if hp < 20:
            status_parts.append(t.t("live_advice.hp_critical", hp=hp))
            action_parts.append(t.t("live_advice.action_hp_critical"))
            tips.append(t.t("live_advice.tip_hp_critical", hp=hp))
        elif hp < 50:
            status_parts.append(t.t("live_advice.hp_low", hp=hp))
            action_parts.append(t.t("live_advice.action_hp_low"))
            tips.append(t.t("live_advice.tip_hp_low"))
        elif hp < 80:
            status_parts.append(t.t("live_advice.hp_moderate", hp=hp))
        else:
            status_parts.append(t.t("live_advice.hp_healthy", hp=hp))

        if armour is not None and armour < 60 and hp > 50:
            tips.append(t.t("live_advice.tip_armour_low", armour=armour))

        # Round state
        if round_active:
            status_parts.append(t.t("live_advice.round_active"))
        else:
            status_parts.append(t.t("live_advice.round_inactive"))
            action_parts.append(t.t("live_advice.action_freeze"))

        # Crosshair
        if crosshair_visible:
            tips.append(t.t("live_advice.tip_crosshair_steady"))
        else:
            tips.append(t.t("live_advice.tip_crosshair_moving"))

        # Kill feed
        if kill_feed_active:
            tips.append(t.t("live_advice.tip_killfeed"))
            if hp < 50:
                action_parts.append(t.t("live_advice.tip_killfeed_cautious"))

        # Money
        if money_visible:
            tips.append(t.t("live_advice.tip_money"))

        # Build result
        status = " | ".join(status_parts) if status_parts else "Analyzing frame..."
        action = (
            " | ".join(action_parts)
            if action_parts
            else t.t("live_advice.action_context_required")
        )
        tips = tips or [t.t("live_advice.tip_default")]

        confidence = 0.7 if len(status_parts) >= 2 else 0.5
        return LiveAdvice(status=status, next_action=action, tips=tips, confidence=confidence)

    @staticmethod
    def _get_num(values: dict, key: str, default: float) -> float:
        val = values.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return default

    @staticmethod
    def _get_optional_num(values: dict, key: str) -> float | None:
        val = values.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return None
