"""Evidence image models for storing and referencing video screenshots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class EvidenceImage(BaseModel):
    """A single evidence screenshot linked to a pipeline event.

    Stores only the file path and metadata rather than raw image bytes
    so the model stays lightweight and serialisable.
    """

    image_id: str
    event_id: str
    frame_index: int
    timestamp_sec: float
    image_path: str
    source: str
    width: int | None = None
    height: int | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def path(self) -> Path:
        return Path(self.image_path)

    def exists(self) -> bool:
        return self.path.exists()
