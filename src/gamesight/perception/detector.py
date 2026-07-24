"""Object-detector contract and YOLO-backed implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import Detection


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: object, frame_index: int, timestamp_sec: float) -> Sequence[Detection]:
        """Return enemy/teammate detections for one decoded frame."""


class YOLODetector(ObjectDetector):
    """Detect players in CS2 frames using a YOLO model (ultralytics).

    Dependency injection: pass *model* for testing with mocks.  In
    production, pass ``None`` and the detector loads a default YOLO
    model via ``ultralytics``.

    The COCO ``person`` class (id 0) is mapped to the label ``"player"``.
    Enemy / teammate classification is handled by a downstream classifier
    (Sprint 4 Task 2).

    Parameters
    ----------
    model:
        A YOLO model object with a ``__call__`` interface (e.g.
        ``ultralytics.YOLO``).  Pass ``None`` to auto-load.
    model_path:
        Path or name of the YOLO model weights.  Used only when *model*
        is ``None``.  Defaults to ``"yolov8n.pt"`` (YOLOv8 nano).
    confidence_threshold:
        Minimum confidence to keep a detection.
    person_class_id:
        COCO class index for ``person`` (default 0).
    """

    _DEFAULT_MODEL = "yolov8n.pt"

    def __init__(
        self,
        model: object | None = None,
        model_path: str | None = None,
        confidence_threshold: float = 0.35,
        person_class_id: int = 0,
    ) -> None:
        self._confidence = confidence_threshold
        self._person_cls = person_class_id

        if model is not None:
            self._model = model
        else:
            self._model = self._load_model(model_path or self._DEFAULT_MODEL)

    def detect(
        self, frame: object, frame_index: int, timestamp_sec: float
    ) -> Sequence[Detection]:
        """Run YOLO inference and return player detections above threshold."""
        if not isinstance(frame, np.ndarray):
            return ()

        image: NDArray[np.uint8] = frame
        results = self._model(image, verbose=False)

        detections: list[Detection] = []
        # results is a list[Results]; take the first (single-image inference)
        result = results[0] if isinstance(results, list) else results

        if result.boxes is None:
            return ()

        for box in result.boxes:
            cls_id = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
            conf = float(box.conf[0]) if hasattr(box.conf, '__getitem__') else float(box.conf)

            if cls_id != self._person_cls or conf < self._confidence:
                continue

            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, '__getitem__') else box.xyxy.tolist()

            detections.append(Detection(
                label="player",
                confidence=conf,
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            ))

        return tuple(detections)

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _load_model(path: str) -> object:
        """Import ultralytics lazily so the module is importable without it."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed.  Install it with: pip install ultralytics"
            ) from exc
        return YOLO(path)