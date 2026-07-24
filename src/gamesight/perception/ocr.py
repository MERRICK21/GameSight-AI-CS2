"""OCR-based HUD text extraction for CS2.

Uses EasyOCR (optional dependency) to read round scores, player names,
and kill-feed entries from the HUD.  All extractors degrade gracefully
when EasyOCR is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


def _get_easyocr():
    """Lazy-import EasyOCR; returns None if not installed."""
    try:
        import easyocr
        return easyocr
    except ImportError:
        return None


class ScoreReader:
    """Read round scores from the round_info HUD region.

    The CS2 round-info area (top-centre) displays the round timer and
    team scores like "CT 3 : 2 T".  This reader extracts the score
    numbers and computes the current round number.

    Round number = CT_score + T_score + 1 (first half is to 12, then switch)

    Usage
    -----
    .. code-block:: python

        reader = ScoreReader()
        ct, t, round_num = reader.read(crop, frame_index, timestamp)
    """

    def __init__(self) -> None:
        self._ocr = None
        self._init_attempted = False
        self._last_scores: tuple[int, int] | None = None
        self._last_round = 0

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            self._ocr = _get_easyocr()
            self._init_attempted = True
            if self._ocr is not None:
                # Lazy init the reader (costly, only once)
                try:
                    self._reader = self._ocr.Reader(["en"], gpu=False, verbose=False)
                except Exception:
                    self._ocr = None
        return self._ocr is not None

    def read(
        self, crop: NDArray[np.uint8], frame_index: int, timestamp_sec: float
    ) -> dict[str, Any]:
        """Return {ct_score, t_score, round_number, active} or defaults."""
        if not self.available:
            return {"round_active": True, "ct_score": 0, "t_score": 0, "round_number": 0}

        try:
            results = self._reader.readtext(crop, detail=0)
            text = " ".join(results).strip()
            # Parse "CT 3 : 2 T" pattern
            nums = []
            for part in text.replace(":", " ").split():
                try:
                    nums.append(int(part))
                except ValueError:
                    pass
            if len(nums) >= 2:
                ct, t = nums[0], nums[1]
                round_num = ct + t + 1
                self._last_scores = (ct, t)
                self._last_round = round_num
                return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": round_num}
        except Exception as exc:
            _log.debug("ScoreReader error: %s", exc)

        # Fallback to last known
        if self._last_scores:
            ct, t = self._last_scores
            return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": self._last_round}
        return {"round_active": True, "ct_score": 0, "t_score": 0, "round_number": 0}


class PlayerNameReader:
    """Read player name from the player_status HUD region.

    When spectating, the bottom-centre area shows the spectated player's
    name.  This reader extracts that name for player-filtering.

    Usage
    -----
    .. code-block:: python

        reader = PlayerNameReader()
        name = reader.read(crop)  # -> str or None
    """

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
                    self._reader = easy.Reader(["en"], gpu=False, verbose=False)
                    self._ocr = easy
                except Exception:
                    pass
        return self._ocr is not None

    def read(self, crop: NDArray[np.uint8]) -> str | None:
        """Return detected player name or None."""
        if not self.available:
            return None
        try:
            results = self._reader.readtext(crop, detail=0)
            if results:
                return results[0].strip()
        except Exception:
            pass
        return None
