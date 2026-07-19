"""Stable domain models for pipeline inputs and outputs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


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
