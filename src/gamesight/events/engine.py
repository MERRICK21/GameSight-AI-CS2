"""Event engine contract combining HUD, detection, and track signals."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from gamesight.domain.models import GameEvent, HudState, Track


class EventEngine(ABC):
    @abstractmethod
    def update(self, hud_state: HudState, tracks: Sequence[Track]) -> Sequence[GameEvent]:
        """Emit newly confirmed temporal events for the current timestamp."""

    @abstractmethod
    def finalize(self) -> Sequence[GameEvent]:
        """Flush pending events once the input video has ended."""
