"""OCR-based round boundary detector using score display.

Reads the round score (e.g., "CT 3 : 2 T") from the HUD and detects
round transitions when the score sum changes.

Compared to the heuristic RoundBoundaryDetector (which watches a
colour-based ``round_active`` boolean), this approach is more robust
because it reads actual game state.

Requires EasyOCR: ``pip install easyocr``.  Falls back gracefully.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import EventType, Evidence, GameEvent, HudState
from gamesight.events.engine import EventEngine
from gamesight.perception.ocr import ScoreReader


class OCRRoundDetector(EventEngine):
    """Detect round boundaries by reading score numbers via OCR.

    Parameters
    ----------
    min_round_duration_sec:
        Suppress implausibly short rounds (default 5s).
    """

    def __init__(self, min_round_duration_sec: float = 5.0) -> None:
        self._reader = ScoreReader()
        self._min_duration = min_round_duration_sec
        self._round_counter = 0
        self._last_round_num = 0
        self._last_start_ts: float | None = None
        self._pending: list[GameEvent] = []

    @property
    def available(self) -> bool:
        return self._reader.available

    def update(
        self, hud_state: HudState, image: NDArray[np.uint8] | None = None, tracks=()
    ) -> Sequence[GameEvent]:
        """Must pass *image* (the full frame) for cropping.  *hud_state* alone is insufficient."""
        self._pending.clear()

        if image is None:
            return ()

        result = self._reader.read(image, hud_state.frame_index, hud_state.timestamp_sec)
        round_num = result.get("round_number", 0)

        if round_num > self._last_round_num:
            # New round detected via score change
            if self._last_start_ts is not None:
                delta = hud_state.timestamp_sec - self._last_start_ts
                if delta >= self._min_duration:
                    # Emit ROUND_END for previous round
                    prev_rid = f"round_{self._round_counter:03d}"
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

            # Start new round
            self._round_counter = round_num
            rid = f"round_{self._round_counter:03d}"
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
        self._round_counter = 0
        self._last_round_num = 0
        self._last_start_ts = None
        self._pending.clear()
        return ()
