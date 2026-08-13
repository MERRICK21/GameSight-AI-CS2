"""Stable domain models for pipeline inputs and outputs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# Python 3.11+ has StrEnum; fallback for older versions.
try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass

from pydantic import BaseModel, Field


class HudRegion(BaseModel):
    """A named, resolution-independent HUD region on screen.

    All coordinates are fractions of screen dimensions (0.0–1.0),
    making profiles reusable across any resolution sharing the same
    aspect ratio.
    """

    name: str
    anchor: str = Field(
        description="One of: top_left, top_center, top_right, center, bottom_left, bottom_center, bottom_right"
    )
    x_norm: float = Field(ge=0.0, le=1.0, description="Left edge as fraction of screen width")
    y_norm: float = Field(ge=0.0, le=1.0, description="Top edge as fraction of screen height")
    w_norm: float = Field(ge=0.0, le=1.0, description="Width as fraction of screen width")
    h_norm: float = Field(ge=0.0, le=1.0, description="Height as fraction of screen height")
    description: str = ""

    def to_pixel(self, screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel bounding box (x, y, w, h)."""
        return (
            int(self.x_norm * screen_w),
            int(self.y_norm * screen_h),
            max(1, int(self.w_norm * screen_w)),
            max(1, int(self.h_norm * screen_h)),
        )


class HudLayoutProfile(BaseModel):
    """Complete screen layout describing where every HUD element lives.

    Profiles are resolution-independent; pixel coordinates are derived
    at runtime via ``HudRegion.to_pixel()``.
    """

    name: str
    game: str
    aspect_ratio: str
    regions: list[HudRegion] = Field(default_factory=list)

    def region(self, name: str) -> HudRegion | None:
        """Look up a region by name; returns None when not found."""
        for r in self.regions:
            if r.name == name:
                return r
        return None

    @property
    def region_names(self) -> list[str]:
        """Convenience accessor for all region names."""
        return [r.name for r in self.regions]


class EventType(StrEnum):
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    PLAYER_KILL = "player_kill"
    PLAYER_DEATH = "player_death"
    ENEMY_FIRST_VISIBLE = "enemy_first_visible"
    COMBAT_START = "combat_start"
    COMBAT_END = "combat_end"
    BOMB_PLANTED = "bomb_planted"
    BOMB_DEFUSED = "bomb_defused"
    KEYFRAME = "keyframe"
    FIRST_PERSON_SUMMARY = "first_person_summary"
    FIRST_PERSON_MOMENT = "first_person_moment"
    ENGAGEMENT_CANDIDATE = "engagement_candidate"


class VideoInput(BaseModel):
    """A user-provided gameplay recording accepted by the ingestion layer."""

    video_id: str
    path: Path
    source_name: str | None = None


class VideoMetadata(BaseModel):
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None


class Evidence(BaseModel):
    """A traceable asset supporting one predicted event or report finding."""

    frame_index: int | None = None
    timestamp_sec: float
    asset_path: str | None = None
    source: str


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    frame_index: int
    timestamp_sec: float


class Track(BaseModel):
    track_id: str
    label: str
    detections: list[Detection] = Field(default_factory=list)


class HudState(BaseModel):
    frame_index: int
    timestamp_sec: float
    profile: str = "unknown"
    values: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class GameEvent(BaseModel):
    event_id: str
    event_type: EventType
    start_sec: float
    end_sec: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RoundAnalysis(BaseModel):
    round_id: str
    start_sec: float
    end_sec: float | None = None
    events: list[GameEvent] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    schema_version: str = "1.0"
    video: VideoInput
    metadata: VideoMetadata
    rounds: list[RoundAnalysis] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)
