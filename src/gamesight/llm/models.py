"""Structured LLM response metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class JsonGenerationResult(BaseModel):
    content: dict[str, Any]
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    usage: LLMUsage = Field(default_factory=LLMUsage)
