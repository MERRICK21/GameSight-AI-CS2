"""Situation-first query planning for the four CS2 knowledge layers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gamesight.knowledge.models import KnowledgeLayer


class DecisionContext(BaseModel):
    """Only observed or user-supplied state; unknown values remain ``None``."""

    side: str | None = None
    alive_teammates: int | None = None
    alive_enemies: int | None = None
    bomb_state: str | None = None
    weapon: str | None = None
    defuse_kit: bool | None = None
    time_remaining_sec: float | None = None
    money: int | None = None
    utility: list[str] = Field(default_factory=list)
    map_name: str | None = None
    map_position: str | None = None

    def query_terms(self) -> list[str]:
        values: list[str] = []
        if self.side:
            values.append(f"side={self.side}")
        if self.alive_teammates is not None and self.alive_enemies is not None:
            values.append(f"situation={self.alive_teammates}v{self.alive_enemies}")
        if self.bomb_state:
            values.append(f"bomb_state={self.bomb_state}")
        if self.weapon:
            values.append(f"weapon={self.weapon}")
        if self.defuse_kit is not None:
            values.append(f"defuse_kit={str(self.defuse_kit).lower()}")
        if self.time_remaining_sec is not None:
            values.append(f"time_remaining={self.time_remaining_sec:.1f}s")
        if self.money is not None:
            values.append(f"money={self.money}")
        if self.utility:
            values.append(f"utility={','.join(self.utility)}")
        if self.map_name:
            values.append(f"map={self.map_name}")
        if self.map_position:
            values.append(f"position={self.map_position}")
        return values


class KnowledgeQueryRouter:
    """Choose an ordered set of layers without turning missing state into facts."""

    _SITUATION_TERMS = (
        "save", "retake", "post-plant", "post plant", "clutch", "low time",
        "alive", "1v", "2v", "3v", "4v", "5v", "保枪", "回防", "残局",
        "人数", "炸弹已安放", "时间不足",
    )
    _DYNAMIC_TERMS = (
        "economy", "price", "reward", "buy", "money", "loss bonus", "cost",
        "经济", "价格", "奖励", "购买", "金钱", "补偿",
    )
    _RULE_TERMS = (
        "win condition", "mechanic", "defuse", "bomb timer", "mr12",
        "胜利条件", "机制", "拆弹", "回合规则",
    )

    def plan(
        self,
        query: str,
        context: DecisionContext | None = None,
    ) -> list[KnowledgeLayer]:
        lowered = query.lower()
        contextual = bool(context and any((
            context.alive_teammates is not None,
            context.alive_enemies is not None,
            context.bomb_state,
            context.time_remaining_sec is not None,
        )))
        situation = contextual or any(term in lowered for term in self._SITUATION_TERMS)
        dynamic = bool(context and context.money is not None) or any(
            term in lowered for term in self._DYNAMIC_TERMS
        )
        rule = any(term in lowered for term in self._RULE_TERMS)

        ordered: list[KnowledgeLayer] = []
        if situation:
            ordered.append(KnowledgeLayer.SITUATION_DECISIONS)
        if dynamic:
            ordered.append(KnowledgeLayer.DYNAMIC_GAME_DATA)
        ordered.append(KnowledgeLayer.TACTICAL_FUNDAMENTALS)
        if rule or not situation:
            ordered.append(KnowledgeLayer.GAME_RULES)
        if KnowledgeLayer.SITUATION_DECISIONS not in ordered:
            ordered.append(KnowledgeLayer.SITUATION_DECISIONS)
        return list(dict.fromkeys(ordered))
