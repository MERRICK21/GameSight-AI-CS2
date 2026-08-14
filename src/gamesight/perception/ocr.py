"""OCR-based HUD text extraction for CS2.

Uses EasyOCR (optional) to read round scores and player names.
Optimized for speed: OCR runs only on sparse keyframes, with
results cached between checks.

Requires: pip install easyocr
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

_OCR_INTERVAL_SEC = 0.75  # Score changes persist; one OCR pass per second is enough.


@dataclass(frozen=True)
class OCRText:
    """One OCR result with a pixel-aligned bounding box."""

    text: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


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
        self._last_ocr_time: float | None = None
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
        if (
            self._last_ocr_time is not None
            and timestamp_sec - self._last_ocr_time < _OCR_INTERVAL_SEC
        ):
            ct, t = self._last_scores
            return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": ct + t + 1}

        self._last_ocr_time = timestamp_sec

        try:
            h, w = crop.shape[:2]
            # Scores occupy fixed boxes below the central timer.  Reading the
            # halves separately prevents the timer (for example 1:55) from
            # being mistaken for a score.
            score_y1, score_y2 = int(h * 0.40), int(h * 0.92)
            left = crop[score_y1:score_y2, int(w * 0.14): int(w * 0.50)]
            right = crop[score_y1:score_y2, int(w * 0.50): int(w * 0.86)]
            left_text = self._reader.readtext(
                left, detail=0, allowlist="0123456789", mag_ratio=4,
                canvas_size=800, text_threshold=0.2, low_text=0.08,
            )
            right_text = self._reader.readtext(
                right, detail=0, allowlist="0123456789", mag_ratio=4,
                canvas_size=800, text_threshold=0.2, low_text=0.08,
            )
            ct = _parse_score_value(left_text)
            t = _parse_score_value(right_text)
            if ct is not None and t is not None:
                self._last_scores = (ct, t)
                self._last_round = ct + t + 1
                return {"round_active": True, "ct_score": ct, "t_score": t, "round_number": self._last_round}
        except Exception:
            pass

        return self._fallback()

    def _fallback(self) -> dict[str, Any]:
        ct, t = self._last_scores
        return {
            "round_active": True,
            "ct_score": ct,
            "t_score": t,
            "round_number": ct + t + 1,
        }


def _parse_score_value(results: list[str]) -> int | None:
    """Convert noisy OCR score tokens into one plausible CS2 score.

    The HUD's vertical separator is commonly recognised as a leading ``7``
    (``71`` for score 1, ``74`` for score 4).  Prefer the first token and
    remove that artefact while retaining legitimate two-digit scores.
    """
    for raw in results:
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        if not digits:
            continue
        if set(digits) == {"0"}:
            return 0
        if len(digits) == 2 and digits.startswith("7"):
            return int(digits[1])
        value = int(digits)
        if 0 <= value <= 30:
            return value
    return None

class PlayerNameReader:
    """Read player name from the player_status HUD region."""

    def __init__(self, languages: tuple[str, ...] = ("ch_sim", "en")) -> None:
        self._ocr = None
        self._init_attempted = False
        self._languages = languages

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            easy = _get_easyocr()
            self._init_attempted = True
            if easy is not None:
                try:
                    self._reader = easy.Reader(list(self._languages), gpu=True, verbose=False)
                    self._ocr = easy
                except Exception:
                    pass
        return self._ocr is not None

    def locate(
        self, crop: NDArray[np.uint8], expected_name: str | None = None
    ) -> OCRText | None:
        """Locate a player name and return its OCR box.

        When ``expected_name`` is supplied, an exact/substring match is
        preferred.  Otherwise the highest-confidence non-numeric label is
        selected; this avoids choosing the adjacent HP/armour numbers.
        """
        if not self.available:
            return None
        try:
            raw_results = self._reader.readtext(crop, detail=1)
        except Exception:
            return None

        detections: list[OCRText] = []
        for box, text, confidence in raw_results:
            cleaned = str(text).strip()
            if not cleaned:
                continue
            xs = [int(round(float(point[0]))) for point in box]
            ys = [int(round(float(point[1]))) for point in box]
            detections.append(OCRText(
                text=cleaned,
                confidence=max(0.0, min(float(confidence), 1.0)),
                bbox_xyxy=(min(xs), min(ys), max(xs), max(ys)),
            ))

        if not detections:
            return None

        if expected_name and expected_name.strip():
            wanted = self._normalise(expected_name)
            matching = [
                item for item in detections
                if wanted in self._normalise(item.text)
                or self._normalise(item.text) in wanted
            ]
            if matching:
                return max(matching, key=lambda item: item.confidence)
            return None

        labels = [
            item for item in detections
            if any(ch.isalpha() or ord(ch) > 127 for ch in item.text)
        ]
        if not labels:
            return None
        return max(labels, key=lambda item: item.confidence)

    def read(
        self, crop: NDArray[np.uint8], expected_name: str | None = None
    ) -> str | None:
        result = self.locate(crop, expected_name)
        return result.text if result is not None else None

    @staticmethod
    def _normalise(value: str) -> str:
        return "".join(ch.casefold() for ch in value if ch.isalnum())


class PlayerStatusValueReader:
    """Read numeric values embedded in the native bottom status HUD.

    This reader is intentionally opt-in.  Neural OCR is appropriate for a
    user-selected screenshot, but is too expensive for every sampled video
    frame.  The current implementation conservatively reads only the armour
    value inside the native shield icon.
    """

    def __init__(self) -> None:
        self._ocr = None
        self._reader = None
        self._init_attempted = False

    @property
    def available(self) -> bool:
        if not self._init_attempted:
            self._init_attempted = True
            easy = _get_easyocr()
            if easy is not None:
                try:
                    self._reader = easy.Reader(["en"], gpu=True, verbose=False)
                    self._ocr = easy
                except Exception:
                    try:
                        self._reader = easy.Reader(["en"], gpu=False, verbose=False)
                        self._ocr = easy
                    except Exception:
                        pass
        return self._ocr is not None and self._reader is not None

    def read_armour(self, player_status_crop: NDArray[np.uint8]) -> int | None:
        """Return native armour points, or ``None`` when not reliable."""
        if not self.available or player_status_crop.size == 0:
            return None
        height, width = player_status_crop.shape[:2]
        # CS2 standard 16:9 player-status region: the shield and its number
        # occupy the lower-left corner.  A tight crop avoids the adjacent HP
        # value while retaining the digits drawn inside the shield.
        shield = player_status_crop[
            int(height * 0.62): int(height * 0.91),
            0: max(1, int(width * 0.052)),
        ]
        if shield.size == 0:
            return None
        try:
            results = self._reader.readtext(
                shield,
                detail=1,
                allowlist="0123456789",
                mag_ratio=4,
                canvas_size=800,
                text_threshold=0.15,
                low_text=0.05,
            )
        except Exception:
            return None
        for _box, raw_text, confidence in results:
            value = _parse_armour_value(str(raw_text), float(confidence))
            if value is not None:
                return value
        return None


def _parse_armour_value(raw_text: str, confidence: float) -> int | None:
    """Parse armour digits while rejecting the shield-outline OCR artefact.

    EasyOCR commonly sees the shield outline as a leading ``6`` (for example
    ``6100`` for 100 armour and ``650`` for 50 armour).  We only accept
    conservative patterns bounded by CS2's 0--100 armour range.
    """
    digits = "".join(ch for ch in raw_text if ch.isdigit())
    if not digits:
        return None
    if "100" in digits:
        return 100
    if len(digits) in (2, 3):
        trailing_two = int(digits[-2:])
        if 10 <= trailing_two <= 99:
            return trailing_two
    if confidence >= 0.40:
        value = int(digits)
        if 0 <= value <= 100:
            return value
    return None
