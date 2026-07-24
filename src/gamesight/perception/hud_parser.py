"""Language-agnostic HUD parsing contract and CS2 implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import HudLayoutProfile, HudState
from gamesight.perception.extractors import RegionExtractor


class HudParser(ABC):
    @abstractmethod
    def identify_profile(self, frame: object) -> str:
        """Identify a HUD layout profile; return 'unknown' when unsupported."""

    @abstractmethod
    def parse(self, frame: object, frame_index: int, timestamp_sec: float) -> HudState:
        """Extract icon/layout-first HUD state for one frame."""


class CS2HudParser(HudParser):
    """CS2 HUD parser that delegates per-region extraction to sub-parsers.

    The parser crops each region defined in *profile* from the frame and
    passes the crop to the corresponding ``RegionExtractor``.  Extracted
    values are merged into a single ``HudState``.

    Extractor key names must match ``HudRegion.name`` values in the
    supplied profile.
    """

    def __init__(
        self,
        profile: HudLayoutProfile,
        extractors: dict[str, RegionExtractor] | None = None,
    ) -> None:
        self._profile = profile
        self._extractors = extractors or {}

    # -- HudParser interface --------------------------------------------------

    def identify_profile(self, frame: object) -> str:
        """Return the configured profile name.

        In this sprint the profile is fixed at construction time.
        Future sprints may inspect actual frame characteristics.
        """
        return self._profile.name

    def parse(
        self, frame: object, frame_index: int, timestamp_sec: float
    ) -> HudState:
        """Crop HUD regions from *frame* and run registered extractors.

        *frame* is expected to be a BGR ``numpy.ndarray`` (e.g. from
        ``VideoFrame.image``).  Regions where no extractor is registered
        are silently skipped.
        """
        if not isinstance(frame, np.ndarray):
            return HudState(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                profile=self._profile.name,
                values={"_error": "frame is not a numpy array"},
                confidence=0.0,
            )

        image: NDArray[np.uint8] = frame
        h, w = image.shape[:2]
        values: dict[str, str | int | float | bool | None] = {}

        for region in self._profile.regions:
            extractor = self._extractors.get(region.name)
            if extractor is None:
                continue

            x, y, rw, rh = region.to_pixel(w, h)
            # Clamp to image bounds to guard against edge-case profiles.
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))

            crop = image[y : y + rh, x : x + rw]
            extracted = extractor.extract(crop, frame_index, timestamp_sec)
            for key, val in extracted.items():
                values[f"{region.name}.{key}"] = val

        return HudState(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            profile=self._profile.name,
            values=values,
            confidence=self._compute_confidence(values),
        )

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _compute_confidence(values: dict[str, object]) -> float:
        """Heuristic confidence based on how many regions produced data."""
        if not values:
            return 0.0
        return min(1.0, 0.5 + 0.1 * len(values))
