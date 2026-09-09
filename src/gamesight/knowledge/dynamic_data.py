"""Patch-sensitive CS2 facts kept separate from durable tactical knowledge."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gamesight.knowledge.models import (
    KnowledgeDocument,
    KnowledgeLayer,
    RuleStrength,
)


class DynamicGameDatum(BaseModel):
    knowledge_id: str
    title: str
    facts: list[str]
    source_urls: list[str] = Field(min_length=1)
    effective_from: str | None = None
    last_verified: str
    notes: list[str] = Field(default_factory=list)

    def to_document(self) -> KnowledgeDocument:
        content = "\n\n".join([
            f"# {self.title}",
            *self.facts,
            *(f"Note: {note}" for note in self.notes),
        ])
        return KnowledgeDocument(
            document_id=f"dynamic_{self.knowledge_id.lower()}",
            title=self.title,
            source_uri=f"builtin://dynamic/{self.knowledge_id}",
            content=content,
            language="en+zh-CN",
            metadata={
                "knowledge_id": self.knowledge_id,
                "knowledge_layer": KnowledgeLayer.DYNAMIC_GAME_DATA.value,
                "rule_strength": RuleStrength.HARD_RULE.value,
                "version_sensitive": True,
                "effective_from": self.effective_from,
                "last_verified": self.last_verified,
                "source_urls": self.source_urls,
            },
        )


# These records deliberately use stable knowledge IDs. Chroma upserts replace
# an old version instead of retaining a stale value under a content-derived ID.
VERSION_SENSITIVE_KNOWLEDGE: tuple[DynamicGameDatum, ...] = (
    DynamicGameDatum(
        knowledge_id="CS2_ECON_KILL_REWARDS",
        title="Competitive kill rewards",
        facts=[
            "Typical rifle and machine-gun kill reward: $300.",
            "AWP kill reward: $100.",
            "Most SMG kill rewards: $600; P90 is a class exception at $300.",
        ],
        source_urls=["https://www.hltv.org/news/38480/how-to-watch-counter-strike"],
        last_verified="2026-08-27",
        notes=["Re-check after gameplay or economy patches."],
    ),
    DynamicGameDatum(
        knowledge_id="CS2_ECON_LOSS_BONUS",
        title="Competitive loss-bonus ladder",
        facts=[
            "Loss compensation uses the values $1400, $1900, $2400, $2900 and $3400.",
        ],
        source_urls=["https://www.hltv.org/news/38480/how-to-watch-counter-strike"],
        last_verified="2026-08-27",
        notes=["The applicable rung depends on match state; do not infer it without scoreboard economy evidence."],
    ),
    DynamicGameDatum(
        knowledge_id="CS2_UTILITY_PRICES",
        title="Competitive grenade prices",
        facts=[
            "Smoke Grenade: $300. Flashbang: $200. HE Grenade: $300.",
            "T Molotov: $400. CT Incendiary Grenade: $500.",
        ],
        source_urls=[
            "local://cs2_basic_rule.docx",
            "https://store.steampowered.com/news/posts/?appids=730&enddate=1719355125&feed=steam_community_announcements",
        ],
        effective_from="2024-05-23",
        last_verified="2026-08-27",
        notes=["The $500 incendiary price is explicitly patch-derived and must remain version-sensitive."],
    ),
    DynamicGameDatum(
        knowledge_id="CS2_T_PLANT_DEFUSED_AWARD",
        title="T team award after planted bomb is defused",
        facts=["The T team award in this outcome is $600."],
        source_urls=[
            "https://store.steampowered.com/news/posts/?appids=730&enddate=1719355125&feed=steam_community_announcements",
        ],
        effective_from="2024-05-23",
        last_verified="2026-08-27",
        notes=["Valve reduced this value from $800 to $600 in the 2024-05-23 update."],
    ),
)


def dynamic_game_documents() -> list[KnowledgeDocument]:
    return [record.to_document() for record in VERSION_SENSITIVE_KNOWLEDGE]


def durable_game_rule_documents() -> list[KnowledgeDocument]:
    """Small official-source facts that are mechanics, not coaching verdicts."""
    return [KnowledgeDocument(
        document_id="official_cs2_responsive_smokes",
        title="CS2 responsive smoke mechanics",
        source_uri="builtin://game_rules/responsive_smokes",
        content=(
            "# Responsive smoke mechanics\n\n"
            "CS2 smoke is a dynamic volumetric object. Bullets and HE grenades "
            "can temporarily push smoke to create a short sightline or expand "
            "occlusion. This mechanic describes what can happen; it does not "
            "prove that shooting a smoke was tactically correct in a given round."
        ),
        language="en",
        metadata={
            "knowledge_id": "CS2_RULE_RESPONSIVE_SMOKE_001",
            "knowledge_layer": KnowledgeLayer.GAME_RULES.value,
            "rule_strength": RuleStrength.HARD_RULE.value,
            "version_sensitive": False,
            "last_verified": "2026-08-27",
            "source_urls": ["https://www.counter-strike.net/cs2"],
        },
    )]
