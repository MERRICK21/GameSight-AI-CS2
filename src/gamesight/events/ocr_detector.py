"""OCR-based round boundary detector using score display.

Reads the round score from the HUD round_info region and detects
round transitions when the score sum changes.

Compared to the heuristic RoundBoundaryDetector (which watches a
colour-based ``round_active`` boolean), this approach is more robust
because it reads actual game state -- but only when fed the correct
cropped region image.

Requires EasyOCR: ``pip install easyocr``.  Falls back gracefully.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import EventType, Evidence, GameEvent, HudLayoutProfile, HudState
from gamesight.events.engine import EventEngine
from gamesight.perception.ocr import ScoreReader

# CS2 competitive max rounds is 30 (MR15 + OT); allow some buffer.
_MAX_ROUND_NUMBER = 60


class OCRRoundDetector(EventEngine):
    """Detect round boundaries by reading score numbers via OCR.

    The detector crops the *round_info* region from the full frame
    using *profile* before running OCR.  This prevents the OCR engine
    from reading unrelated on-screen text (kill feed, chat, weapon
    names) that would produce garbage round numbers.

    Parameters
    ----------
    profile:
        HudLayoutProfile used to locate the round_info region.
    min_round_duration_sec:
        Suppress implausibly short rounds (default 5s).
    """

    def __init__(
        self,
        profile: HudLayoutProfile | None = None,
        min_round_duration_sec: float = 5.0,
    ) -> None:
        self._reader = ScoreReader()
        self._profile = profile
        self._min_duration = min_round_duration_sec
        self._seq_counter = 0       # sequential round counter (1-based)
        self._last_round_num = 0
        self._last_start_ts: float | None = None
        self._pending: list[GameEvent] = []

    @property
    def available(self) -> bool:
        return self._reader.available

    def update(
        self, hud_state: HudState, image: NDArray[np.uint8] | None = None, tracks=()
    ) -> Sequence[GameEvent]:
        """Must pass *image* (the full frame).  The detector crops the
        round_info region internally before calling OCR.

        *hud_state* alone is insufficient because OCR needs pixel data.
        """
        self._pending.clear()

        if image is None:
            return ()

        # Crop to round_info region so OCR only sees the scores.
        crop = self._crop_round_info(image)
        if crop is None:
            return ()

        result = self._reader.read(crop, hud_state.frame_index, hud_state.timestamp_sec)
        round_num = result.get("round_number", 0)

        # Clamp to plausible CS2 range.
        round_num = max(1, min(round_num, _MAX_ROUND_NUMBER))

        if round_num > self._last_round_num:
            # New round detected via score change.
            if self._last_start_ts is not None:
                delta = hud_state.timestamp_sec - self._last_start_ts
                if delta >= self._min_duration:
                    # Emit ROUND_END for previous round.
                    prev_rid = f"round_{self._seq_counter:03d}"
                    self._pending.append(GameEvent(
                        event_id=f"round_end_{prev_rid}",
                        event_type=EventType.ROUND_END,
                        start_sec=hud_state.timestamp_sec,
                        confidence=0.95,
                        evidence=[Evidence(
                            frame_index=hud_state.frame_index,
                            timestamp_sec=hud_state.timestamp_sec,
                            source="OCRRoundDetector.score_change",
                        )],
                        attributes={"round_id": prev_rid},
                    ))

            # Start new round -- use sequential counter, not raw OCR value.
            self._seq_counter += 1
            rid = f"round_{self._seq_counter:03d}"
            self._last_start_ts = hud_state.timestamp_sec
            self._last_round_num = round_num
            self._pending.append(GameEvent(
                event_id=f"round_start_{rid}",
                event_type=EventType.ROUND_START,
                start_sec=hud_state.timestamp_sec,
                confidence=0.95,
                evidence=[Evidence(
                    frame_index=hud_state.frame_index,
                    timestamp_sec=hud_state.timestamp_sec,
                    source="OCRRoundDetector.score_change",
                )],
                attributes={
                    "round_id": rid,
                    "ct_score": result.get("ct_score", 0),
                    "t_score": result.get("t_score", 0),
                },
            ))

        return tuple(self._pending)

    def finalize(self) -> Sequence[GameEvent]:
        self._seq_counter = 0
        self._last_round_num = 0
        self._last_start_ts = None
        self._pending.clear()
        return ()

    # -- internal ------------------------------------------------------------

    def _crop_round_info(self, image: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        """Crop the round_info region from the full frame.

        Falls back to a heuristic central-top strip when no profile is
        configured.
        """
        h, w = image.shape[:2]
        if self._profile is not None:
            region = self._profile.region("round_info")
            if region is not None:
                x, y, rw, rh = region.to_pixel(w, h)
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                rw = max(1, min(rw, w - x))
                rh = max(1, min(rh, h - y))
                return image[y: y + rh, x: x + rw]

        # Fallback: central top 30% x 10% strip (common placement).
        x = int(w * 0.32)
        y = int(h * 0.005)
        rw = int(w * 0.36)
        rh = int(h * 0.10)
        return image[y: y + rh, x: x + rw]
