"""Tracker contract and IOU-based multi-object tracker implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import Detection, Track


class MultiObjectTracker(ABC):
    @abstractmethod
    def update(self, detections: Sequence[Detection]) -> Sequence[Track]:
        """Associate current detections with persistent object trajectories."""

    @abstractmethod
    def reset(self) -> None:
        """Clear tracker state at a video or round boundary."""


class IOUTracker(MultiObjectTracker):
    """IOU-based multi-object tracker.

    Matches detections frame-to-frame using Intersection-over-Union.
    New detections that do not match any active track spawn new tracks.
    Tracks that are not matched for *max_lost_frames* consecutive frames
    are terminated.

    Parameters
    ----------
    iou_threshold:
        Minimum IOU for a detection-track pair to be considered a match.
    max_lost_frames:
        Number of consecutive unmatched frames before a track is dropped.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_lost_frames < 1:
            raise ValueError("max_lost_frames must be >= 1")

        self._iou_threshold = iou_threshold
        self._max_lost = max_lost_frames

        self._tracks: dict[str, Track] = {}
        self._lost_count: dict[str, int] = {}
        self._next_id = 0

    # -- MultiObjectTracker interface ----------------------------------------

    def update(self, detections: Sequence[Detection]) -> Sequence[Track]:
        """Run one tracking iteration.

        Returns a snapshot of all currently active tracks after this frame.
        """
        det_list = list(detections)
        active_ids = list(self._tracks.keys())

        if not det_list:
            for tid in active_ids:
                self._lost_count[tid] = self._lost_count.get(tid, 0) + 1
            self._prune()
            return tuple(self._tracks.values())

        if not active_ids:
            for det in det_list:
                self._create_track(det)
            return tuple(self._tracks.values())

        # Compute IOU matrix (tracks x detections).
        iou = self._compute_iou_matrix(active_ids, det_list)

        # Greedy matching.
        matched_track_ids: set[str] = set()
        matched_det_indices: set[int] = set()

        while True:
            best_val = -1.0
            best = (-1, -1)
            for ti, tid in enumerate(active_ids):
                if tid in matched_track_ids:
                    continue
                for di in range(len(det_list)):
                    if di in matched_det_indices:
                        continue
                    if iou[ti, di] > best_val:
                        best_val = iou[ti, di]
                        best = (ti, di)

            if best_val < self._iou_threshold:
                break

            ti, di = best
            tid = active_ids[ti]
            det = det_list[di]

            self._tracks[tid].detections.append(det)
            self._lost_count[tid] = 0
            matched_track_ids.add(tid)
            matched_det_indices.add(di)

        # Unmatched tracks: increment lost counter.
        for tid in active_ids:
            if tid not in matched_track_ids:
                self._lost_count[tid] = self._lost_count.get(tid, 0) + 1

        # Unmatched detections: create new tracks.
        for di, det in enumerate(det_list):
            if di not in matched_det_indices:
                self._create_track(det)

        self._prune()
        return tuple(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._lost_count.clear()
        self._next_id = 0

    # -- internal ------------------------------------------------------------

    def _create_track(self, det: Detection) -> Track:
        tid = f"track_{self._next_id:04d}"
        self._next_id += 1
        track = Track(track_id=tid, label=det.label, detections=[det])
        self._tracks[tid] = track
        self._lost_count[tid] = 0
        return track

    def _prune(self) -> None:
        expired = [
            tid for tid, count in self._lost_count.items()
            if count >= self._max_lost
        ]
        for tid in expired:
            self._tracks.pop(tid, None)
            self._lost_count.pop(tid, None)

    def _compute_iou_matrix(
        self,
        track_ids: list[str],
        detections: list[Detection],
    ) -> NDArray[np.float64]:
        n_tracks = len(track_ids)
        n_dets = len(detections)
        iou = np.zeros((n_tracks, n_dets), dtype=np.float64)

        for ti, tid in enumerate(track_ids):
            last_det = self._tracks[tid].detections[-1]
            a_box = last_det.bbox_xyxy
            for di, det in enumerate(detections):
                iou[ti, di] = _iou(a_box, det.bbox_xyxy)

        return iou


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two axis-aligned bounding boxes (xyxy)."""
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0
