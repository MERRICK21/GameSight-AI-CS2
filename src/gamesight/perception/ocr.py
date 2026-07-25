"""OCR-based HUD text extraction for CS2.

Uses EasyOCR (optional) to read round scores and player names.
Optimized for speed: OCR runs only on sparse keyframes, with
results cached between checks.

Requires: pip install easyocr
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

_OCR_INTERVAL_SEC = 3.0  # Only run OCR every N seconds


def _get_easyocr():
    try:
        import easyocr
        return easyocr
    except ImportError:
        return None


class ScoreReader:
    """Read round scores from the round_info HUD region.

    To avoid running OCR on every frame (which is prohibitively slow),
    this reader only performs OCR every ``_OCR_INTERVAL_SEC`` seconds.
    Between OCR checks, cached values are returned.
    """

    def __init__(self) -> None:
        self._ocr = None
        self._init_attempted = False
        self._last_ocr_time = 0.0
        self._last_scores: tuple[int, int] = (0, 0)
        self._last_round = 0

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            easy = _get_easyocr()
            self._init_attempted = True
            if easy is not None:
                try:
                    self._reader = easy.Reader(["en"], gpu=True, verbose=False)
                    self._ocr = easy
                except Exception:
                    pass
        return self._ocr is not None

    def read(
        self, crop: NDArray[np.uint8], frame_index: int, timestamp_sec: float
    ) -> dict[str, Any]:
        """Return {ct_score, t_score, round_number, active}. Runs OCR only every N seconds."""
        if not self.available:
            return self._fallback()

        # Only run OCR every _OCR_INTERVAL_SEC seconds
        if timestamp_sec - self._last_ocr_time < _OCR_INTERVAL_SEC:
            ct, t = self._last_scores
            return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": ct + t + 1}

        self._last_ocr_time = timestamp_sec

        try:
            results = self._reader.readtext(crop, detail=0)
            text = " ".join(results).strip()
            nums = []
            for part in text.replace(":", " ").split():
                try:
                    nums.append(int(part))
                except ValueError:
                    pass
            if len(nums) >= 2:
                ct, t = nums[0], nums[1]
                self._last_scores = (ct, t)
                self._last_round = ct + t + 1
                return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": self._last_round}
        except Exception:
            pass

        return self._fallback()

    def _fallback(self) -> dict[str, Any]:
        ct, t = self._last_scores
        return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": ct + t + 1}


class PlayerNameReader:
    """Read player name from the player_status HUD region."""

    def __init__(self) -> None:
        self._ocr = None
        self._init_attempted = False

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            easy = _get_easyocr()
            self._init_attempted = True
            if easy is not None:
                try:
                    self._reader = easy.Reader(["en"], gpu=True, verbose=False)
                    self._ocr = easy
                except Exception:
                    pass
        return self._ocr is not None

    def read(self, crop: NDArray[np.uint8]) -> str | None:
        if not self.available:
            return None
        try:
            results = self._reader.readtext(crop, detail=0)
            if results:
                return results[0].strip()
        except Exception:
            pass
        return None
