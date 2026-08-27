"""Deterministic, structure-aware chunking for CS2 knowledge sources."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

from gamesight.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeLayer,
    RuleStrength,
)


_NUMBERED_HEADING = re.compile(r"^(?:\d+[.)]|\d+(?:\.\d+)+)\s+\S")
_SECTION_NUMBER = re.compile(r"^(\d+)(?:[.)]|\.\d+)")
_METADATA_LINE = re.compile(r"^([a-z_]+)\s*:\s*(.+)$", re.IGNORECASE)


class KnowledgeChunker:
    """Split sources into compact passages with stable IDs and small overlap.

    The project manual uses ordinary DOCX paragraphs rather than Word heading
    styles.  Numbered section lines are therefore promoted to headings before
    chunking so that economy facts cannot silently share a chunk with tactical
    guidance or situation decisions.
    """

    def __init__(
        self,
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
        min_chars: int = 40,
    ) -> None:
        if max_chars < 200:
            raise ValueError("max_chars must be at least 200")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be between 0 and max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        passages: list[tuple[str | None, str]] = []
        heading: str | None = None
        buffer = ""

        def flush() -> None:
            nonlocal buffer
            text = buffer.strip()
            if text:
                passages.extend(self._window(heading, text))
            buffer = ""

        for block in document.content.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            first_line = lines[0].strip()
            is_markdown_heading = first_line.startswith("#")
            is_numbered_heading = self._is_numbered_heading(first_line)
            if is_markdown_heading or is_numbered_heading:
                flush()
                heading = (
                    first_line.lstrip("#").strip()
                    if is_markdown_heading else first_line
                ) or heading
                remainder = "\n".join(lines[1:]).strip()
                if remainder:
                    buffer = remainder
                continue
            candidate = f"{buffer}\n\n{block}".strip() if buffer else block
            if len(candidate) <= self.max_chars:
                buffer = candidate
            else:
                flush()
                buffer = block
        flush()

        chunks: list[KnowledgeChunk] = []
        for index, (chunk_heading, content) in enumerate(passages):
            if len(content) < self.min_chars and chunks:
                previous = chunks[-1]
                merged = f"{previous.content}\n\n{content}".strip()
                if (
                    previous.heading == chunk_heading
                    and len(merged) <= self.max_chars + self.overlap_chars
                ):
                    chunks[-1] = previous.model_copy(update={"content": merged})
                    continue
            annotations = self._annotations(document, chunk_heading, content)
            stable_key = document.metadata.get("knowledge_id")
            digest_input = (
                f"{stable_key}:{index}"
                if stable_key
                else f"{document.document_id}:{index}:{content}"
            )
            digest = sha256(digest_input.encode("utf-8")).hexdigest()[:20]
            chunks.append(KnowledgeChunk(
                chunk_id=f"chunk_{digest}",
                document_id=document.document_id,
                title=document.title,
                source_uri=document.source_uri,
                content=content,
                chunk_index=index,
                heading=chunk_heading,
                language=document.language,
                metadata=document.metadata.copy(),
                **annotations,
            ))
        return chunks

    @staticmethod
    def _is_numbered_heading(line: str) -> bool:
        if not _NUMBERED_HEADING.match(line):
            return False
        # Long prose sentences that happen to begin with a number are content.
        return len(line) <= 180 and line.count("。") + line.count(".") <= 4

    def _annotations(
        self,
        document: KnowledgeDocument,
        heading: str | None,
        content: str,
    ) -> dict[str, Any]:
        metadata = document.metadata
        joined = f"{heading or ''}\n{content}".lower()
        layer = self._enum_value(
            KnowledgeLayer,
            metadata.get("knowledge_layer"),
            self._infer_layer(heading, joined),
        )
        default_strength = {
            KnowledgeLayer.GAME_RULES: RuleStrength.HARD_RULE,
            KnowledgeLayer.DYNAMIC_GAME_DATA: RuleStrength.HARD_RULE,
            KnowledgeLayer.SITUATION_DECISIONS: RuleStrength.CONTEXTUAL_RECOMMENDATION,
            KnowledgeLayer.TACTICAL_FUNDAMENTALS: RuleStrength.STRATEGIC_PRINCIPLE,
        }[layer]
        rule_strength = self._enum_value(
            RuleStrength, metadata.get("rule_strength"), default_strength,
        )
        inline = self._inline_metadata(content)
        version_sensitive = self._as_bool(
            metadata.get("version_sensitive", inline.get("version_sensitive", False))
        )
        if layer == KnowledgeLayer.DYNAMIC_GAME_DATA:
            version_sensitive = True
        if version_sensitive:
            layer = KnowledgeLayer.DYNAMIC_GAME_DATA
            rule_strength = RuleStrength.HARD_RULE
        exceptions = metadata.get("exceptions") or inline.get("exceptions") or []
        if isinstance(exceptions, str):
            exceptions = [item.strip() for item in re.split(r"[;；]", exceptions) if item.strip()]
        urls = metadata.get("source_urls") or []
        if isinstance(urls, str):
            urls = [urls]
        return {
            "layer": layer,
            "rule_strength": rule_strength,
            "version_sensitive": version_sensitive,
            "last_verified": metadata.get("last_verified") or inline.get("last_verified"),
            "effective_from": metadata.get("effective_from") or inline.get("effective_from"),
            "expires_at": metadata.get("expires_at") or inline.get("expires_at"),
            "source_urls": list(urls),
            "exceptions": list(exceptions),
        }

    @staticmethod
    def _infer_layer(heading: str | None, text: str) -> KnowledgeLayer:
        section_match = _SECTION_NUMBER.match(heading or "")
        if section_match:
            section = int(section_match.group(1))
            if 1 <= section <= 5:
                return KnowledgeLayer.GAME_RULES
            if 36 <= section <= 40 or section == 48:
                return KnowledgeLayer.DYNAMIC_GAME_DATA
            if 49 <= section <= 85 or section == 96:
                return KnowledgeLayer.SITUATION_DECISIONS
            # The remaining numbered manual sections describe general tactics,
            # terminology or evaluation methodology unless inline metadata below
            # explicitly promotes them to dynamic data.
            return KnowledgeLayer.TACTICAL_FUNDAMENTALS
        dynamic_terms = (
            "version_sensitive", "starting money", "kill reward", "loss bonus",
            "weapon price", "grenade price", "purchase price", "defuse kit", "经济数字", "价格",
            "奖励金额", "失败补偿",
        )
        situation_terms = (
            "situation decision", "save decision", "post-plant", "post plant",
            "retake", "clutch", "low time", "late-round", "late round",
            "残局", "回防", "保枪", "人数劣势", "人数优势", "低时间",
        )
        game_rule_terms = (
            "hard rule", "victory condition", "round phase", "bomb mechanic",
            "defuse", "mr12", "胜利条件", "回合规则", "炸弹机制", "拆弹",
        )
        if any(term in text for term in dynamic_terms):
            return KnowledgeLayer.DYNAMIC_GAME_DATA
        if any(term in text for term in situation_terms):
            return KnowledgeLayer.SITUATION_DECISIONS
        if any(term in text for term in game_rule_terms):
            return KnowledgeLayer.GAME_RULES
        return KnowledgeLayer.TACTICAL_FUNDAMENTALS

    @staticmethod
    def _inline_metadata(content: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in content.splitlines():
            match = _METADATA_LINE.match(line.strip())
            if match:
                values[match.group(1).lower()] = match.group(2).strip()
        return values

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "yes", "1"}

    @staticmethod
    def _enum_value(enum_type, value: Any, default):
        try:
            return enum_type(value) if value is not None else default
        except ValueError:
            return default

    def _window(self, heading: str | None, text: str) -> list[tuple[str | None, str]]:
        if len(text) <= self.max_chars:
            return [(heading, text)]
        windows: list[tuple[str | None, str]] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            if end < len(text):
                boundary = max(
                    text.rfind("。", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("\n", start, end),
                )
                if boundary > start + self.max_chars // 2:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                windows.append((heading, piece))
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap_chars)
        return windows
