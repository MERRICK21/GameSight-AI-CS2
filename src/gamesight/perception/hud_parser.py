"""Language-agnostic HUD parsing contract."""

from abc import ABC, abstractmethod

from gamesight.domain.models import HudState


class HudParser(ABC):
    @abstractmethod
    def identify_profile(self, frame: object) -> str:
        """Identify a HUD layout profile; return 'unknown' when unsupported."""

    @abstractmethod
    def parse(self, frame: object, frame_index: int, timestamp_sec: float) -> HudState:
        """Extract icon/layout-first HUD state for one frame."""
