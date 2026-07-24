"""Player classifier: enemy vs teammate via colour analysis of bbox crops."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import Detection


class PlayerClassifier:
    """Classify player detections as ``"enemy"`` or ``"teammate"`` by
    analysing the colour distribution inside each bounding-box crop.

    CS2 renders enemies with a red outline and teammates with a blue
    outline.  This classifier counts red- and blue-range pixels in BGR
    space and picks the dominant colour when it exceeds a configurable
    ratio.

    Detections whose crops produce no strong colour signal keep the
    original ``"player"`` label.

    Parameters
    ----------
    red_low / red_high:
        BGR lower / upper bounds for enemy red.
    blue_low / blue_high:
        BGR lower / upper bounds for teammate blue.
    min_ratio:
        Minimum ratio of dominant-colour pixels over the other colour
        required to classify.  Default 1.5 means the winner must have
        at least 50 % more pixels than the runner-up.
    edge_margin:
        Fraction of the crop to trim from each side before sampling
        (focuses on the player silhouette rather than surroundings).
    """

    def __init__(
        self,
        red_low: tuple[int, int, int] = (0, 0, 160),
        red_high: tuple[int, int, int] = (70, 90, 255),
        blue_low: tuple[int, int, int] = (120, 0, 0),
        blue_high: tuple[int, int, int] = (255, 180, 80),
        min_ratio: float = 1.5,
        edge_margin: float = 0.05,
    ) -> None:
        self._red_low = np.array(red_low, dtype=np.uint8)
        self._red_high = np.array(red_high, dtype=np.uint8)
        self._blue_low = np.array(blue_low, dtype=np.uint8)
        self._blue_high = np.array(blue_high, dtype=np.uint8)
        self._min_ratio = min_ratio
        self._edge_margin = edge_margin

    def classify(
        self,
        frame: NDArray[np.uint8],
        detections: Sequence[Detection],
    ) -> list[Detection]:
        """Return a new list of detections with labels updated to
        ``"enemy"``, ``"teammate"``, or left as ``"player"``.
        """
        classified: list[Detection] = []
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            x1_i = max(0, int(x1))
            y1_i = max(0, int(y1))
            x2_i = min(w, int(x2))
            y2_i = min(h, int(y2))

            if x2_i <= x1_i or y2_i <= y1_i:
                classified.append(det)
                continue

            crop = frame[y1_i:y2_i, x1_i:x2_i]
            if crop.size == 0:
                classified.append(det)
                continue

            # Trim edges to focus on the player silhouette.
            ch, cw = crop.shape[:2]
            mx = int(cw * self._edge_margin)
            my = int(ch * self._edge_margin)
            core = crop[my : ch - my, mx : cw - mx] if ch > 2 * my and cw > 2 * mx else crop

            red_count = int(np.sum(_cv_in_range(core, self._red_low, self._red_high)))
            blue_count = int(np.sum(_cv_in_range(core, self._blue_low, self._blue_high)))

            label = det.label
            if red_count > 0 and blue_count > 0:
                if red_count / blue_count >= self._min_ratio:
                    label = "enemy"
                elif blue_count / red_count >= self._min_ratio:
                    label = "teammate"
            elif red_count > 0:
                label = "enemy"
            elif blue_count > 0:
                label = "teammate"

            classified.append(Detection(
                label=label,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                frame_index=det.frame_index,
                timestamp_sec=det.timestamp_sec,
            ))

        return classified


def _cv_in_range(
    image: NDArray[np.uint8],
    lower: NDArray[np.uint8],
    upper: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """NumPy ``cv2.inRange`` equivalent."""
    if image.size == 0:
        return np.array([], dtype=np.uint8)
    if image.ndim == 2:
        return ((image >= lower[0]) & (image <= upper[0])).astype(np.uint8) * 255
    return np.all((image >= lower) & (image <= upper), axis=-1).astype(np.uint8) * 255
