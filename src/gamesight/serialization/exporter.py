"""JSON exporter for MatchTimeline — serialisation with evidence linking."""

from __future__ import annotations

import json
from pathlib import Path

from gamesight.serialization.timeline import MatchTimeline, RoundTimeline, TimelineEvent


class TimelineExporter:
    """Serialise a MatchTimeline to JSON with human-readable formatting.

    By default the exporter uses ``indent=2``, ``ensure_ascii=False``,
    and serialises Pydantic models via ``.model_dump(mode='json')`` so
    that ``Path``, ``datetime``, and enum values are coerced correctly.
    """

    def __init__(self, indent: int = 2, ensure_ascii: bool = False) -> None:
        self._indent = indent
        self._ensure_ascii = ensure_ascii

    def to_json(self, timeline: MatchTimeline) -> str:
        """Return the MatchTimeline as a JSON string."""
        return json.dumps(
            timeline.model_dump(mode="json"),
            indent=self._indent,
            ensure_ascii=self._ensure_ascii,
        )

    def export(self, timeline: MatchTimeline, path: Path | str) -> Path:
        """Write the MatchTimeline to a JSON file.

        Returns the resolved ``Path`` of the written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(timeline), encoding="utf-8")
        return out.resolve()


def timeline_to_json(timeline: MatchTimeline, indent: int = 2) -> str:
    """Convenience function: serialise a MatchTimeline to a JSON string."""
    return TimelineExporter(indent=indent).to_json(timeline)


def export_timeline(
    timeline: MatchTimeline,
    path: Path | str,
    indent: int = 2,
) -> Path:
    """Convenience function: export a MatchTimeline to a JSON file."""
    return TimelineExporter(indent=indent).export(timeline, path)
