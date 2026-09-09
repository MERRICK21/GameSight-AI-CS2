"""Build explicit known/unknown coaching context from native HUD evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher

from gamesight.domain.models import (
    ContextObservation, RoundAnalysis, RoundContextEvidence,
)


def build_round_contexts(
    rounds: Sequence[RoundAnalysis],
    samples: Sequence[object],
    clock_observations: Sequence[ContextObservation | dict] = (),
    money_observations: Sequence[ContextObservation | dict] = (),
    position_observations: Sequence[ContextObservation | dict] = (),
) -> list[RoundContextEvidence]:
    normalized_positions = [
        normalized
        for observation in position_observations
        if (normalized := _normalize_position_observation(
            _as_observation(observation)
        )) is not None
    ]
    match_map, map_confidence = _resolve_map(normalized_positions)
    contexts: list[RoundContextEvidence] = []
    for round_analysis in rounds:
        end_sec = round_analysis.end_sec
        round_samples = [
            sample for sample in samples
            if float(getattr(sample, "timestamp_sec")) >= round_analysis.start_sec
            and (end_sec is None or float(getattr(sample, "timestamp_sec")) <= end_sec)
        ]
        sides = [
            str(getattr(sample, "player_team"))
            for sample in round_samples
            if getattr(sample, "player_team", None) in {"t", "ct"}
        ]
        side = None
        confidence = 0.0
        sources: dict[str, str] = {}
        if sides:
            side, count = Counter(sides).most_common(1)[0]
            confidence = count / len(sides)
            sources["player_side"] = "FirstPersonAnalyzer.native_team_emblem"
        clock_candidates = [
            _as_observation(observation)
            for observation in clock_observations
            if _in_round(
                float(_observation_field(observation, "timestamp_sec")),
                round_analysis,
            )
        ]
        verified_clocks = _verified_clock_chain(clock_candidates)
        native_clock = (
            float(verified_clocks[0].value) if verified_clocks else None
        )
        if verified_clocks:
            sources["native_round_clock"] = (
                "OCRRoundDetector.native_round_clock.continuity_verified"
            )

        weapon_candidates = [
            ContextObservation(
                frame_index=getattr(sample, "frame_index", None),
                timestamp_sec=float(getattr(sample, "timestamp_sec")),
                value=str(getattr(sample, "weapon")),
                confidence=float(getattr(sample, "weapon_confidence", 0.0)),
                source=str(getattr(sample, "weapon_source", "")),
            )
            for sample in round_samples
            if getattr(sample, "weapon", None) is not None
            and getattr(sample, "weapon_source", None)
        ]
        verified_weapons = _verified_weapon_observations(weapon_candidates)
        weapon_categories = sorted({str(item.value) for item in verified_weapons})
        weapon = None
        if verified_weapons:
            weighted = Counter()
            for item in verified_weapons:
                weighted[str(item.value)] += item.confidence
            weapon = weighted.most_common(1)[0][0]
            sources["weapon"] = "FirstPersonWeaponClassifier.held_view"

        money_candidates = [
            _as_observation(observation)
            for observation in money_observations
            if _in_round(
                float(_observation_field(observation, "timestamp_sec")),
                round_analysis,
            )
        ]
        verified_money = _verified_money_observations(money_candidates)
        money = int(verified_money[0].value) if verified_money else None
        if verified_money:
            sources["money"] = "OCRRoundDetector.native_money.stability_verified"

        utility_candidates = [
            ContextObservation(
                frame_index=getattr(sample, "frame_index", None),
                timestamp_sec=float(getattr(sample, "timestamp_sec")),
                value=str(getattr(sample, "held_utility")),
                confidence=float(getattr(sample, "utility_confidence", 0.0)),
                source=str(getattr(sample, "utility_source", "")),
            )
            for sample in round_samples
            if getattr(sample, "held_utility", None) is not None
            and getattr(sample, "utility_source", None)
        ]
        verified_utility = _verified_utility_observations(utility_candidates)
        utility = sorted({str(item.value) for item in verified_utility})
        if verified_utility:
            sources["utility"] = "FirstPersonUtilityClassifier.held_view"

        position_candidates = [
            observation for observation in normalized_positions
            if _in_round(observation.timestamp_sec, round_analysis)
        ]
        verified_positions = _verified_position_observations(position_candidates)
        map_position = (
            str(verified_positions[0].value) if verified_positions else None
        )
        if verified_positions:
            sources["map_position"] = (
                "OCRRoundDetector.native_location_text.vocabulary_verified"
            )
        map_name = match_map
        if map_name is not None:
            sources["map_name"] = "NativeLocationReader.match_vocabulary"
        contexts.append(RoundContextEvidence(
            round_id=round_analysis.round_id,
            player_side=side,
            player_side_confidence=confidence,
            native_round_clock_sec=native_clock,
            round_clock_observations=verified_clocks,
            weapon=weapon,
            weapon_categories=weapon_categories,
            weapon_observations=verified_weapons,
            money=money,
            money_observations=verified_money,
            utility=utility,
            utility_observations=verified_utility,
            map_name=map_name,
            map_name_confidence=map_confidence if map_name else 0.0,
            map_position=map_position,
            position_observations=verified_positions,
            sources=sources,
        ))
    return contexts


def context_coverage(contexts: Sequence[RoundContextEvidence]) -> dict[str, int]:
    return {
        "rounds": len(contexts),
        "player_side": sum(context.player_side is not None for context in contexts),
        "native_round_clock": sum(
            context.native_round_clock_sec is not None for context in contexts
        ),
        "weapon": sum(context.weapon is not None for context in contexts),
        "economy": sum(context.money is not None for context in contexts),
        "utility": sum(bool(context.utility) for context in contexts),
        "map_position": sum(
            context.map_name is not None and context.map_position is not None
            for context in contexts
        ),
    }


def _observation_field(observation: ContextObservation | dict, field: str):
    if isinstance(observation, ContextObservation):
        return getattr(observation, field)
    return observation[field]


def _as_observation(observation: ContextObservation | dict) -> ContextObservation:
    if isinstance(observation, ContextObservation):
        return observation
    return ContextObservation.model_validate(observation)


def _in_round(timestamp_sec: float, round_analysis: RoundAnalysis) -> bool:
    return timestamp_sec >= round_analysis.start_sec and (
        round_analysis.end_sec is None or timestamp_sec <= round_analysis.end_sec
    )


def _verified_clock_chain(
    observations: Sequence[ContextObservation],
) -> list[ContextObservation]:
    """Keep a native clock only when consecutive OCR reads are time-consistent."""
    candidates = sorted(
        (item for item in observations if item.confidence >= .35),
        key=lambda item: item.timestamp_sec,
    )
    best: list[ContextObservation] = []
    current: list[ContextObservation] = []
    for item in candidates:
        if not isinstance(item.value, (int, float)):
            continue
        if not current:
            current = [item]
            continue
        previous = current[-1]
        elapsed = item.timestamp_sec - previous.timestamp_sec
        clock_drop = float(previous.value) - float(item.value)
        if elapsed > 0 and -1.0 <= clock_drop <= elapsed + 4.0:
            current.append(item)
        else:
            if _clock_chain_is_verified(current) and len(current) > len(best):
                best = current
            current = [item]
    if _clock_chain_is_verified(current) and len(current) > len(best):
        best = current
    return best


def _clock_chain_is_verified(chain: Sequence[ContextObservation]) -> bool:
    return len(chain) >= 2 and any(
        float(previous.value) - float(current.value) >= 1.0
        for previous, current in zip(chain, chain[1:])
    )


def _verified_weapon_observations(
    observations: Sequence[ContextObservation],
) -> list[ContextObservation]:
    """Require repeat evidence, except for the specific native scope signal."""
    by_category: dict[str, list[ContextObservation]] = {}
    for item in observations:
        if item.confidence < .80:
            continue
        by_category.setdefault(str(item.value), []).append(item)
    verified: list[ContextObservation] = []
    for category, items in by_category.items():
        if len(items) >= 2 or any(
            category in {"sniper", "c4"}
            and item.confidence >= .95
            and (
                "native_scope_view" in item.source
                or "held_c4_geometry" in item.source
            )
            for item in items
        ):
            verified.extend(items)
    return sorted(verified, key=lambda item: item.timestamp_sec)


def _verified_money_observations(
    observations: Sequence[ContextObservation],
) -> list[ContextObservation]:
    candidates = sorted(
        (
            item for item in observations
            if item.confidence >= .55
            and isinstance(item.value, (int, float))
            and 0 <= int(item.value) <= 16000
        ),
        key=lambda item: item.timestamp_sec,
    )
    chains: list[list[ContextObservation]] = []
    current: list[ContextObservation] = []
    for item in candidates:
        if (
            current
            and (
                int(item.value) != int(current[-1].value)
                or item.timestamp_sec - current[-1].timestamp_sec > 10.5
            )
        ):
            if len(current) >= 2:
                chains.append(current)
            current = []
        current.append(item)
    if len(current) >= 2:
        chains.append(current)
    if not chains:
        return []
    return min(chains, key=lambda chain: chain[0].timestamp_sec)


def _verified_utility_observations(
    observations: Sequence[ContextObservation],
) -> list[ContextObservation]:
    by_category: dict[str, list[ContextObservation]] = {}
    for item in observations:
        if item.confidence >= .85:
            by_category.setdefault(str(item.value), []).append(item)
    verified: list[ContextObservation] = []
    for items in by_category.values():
        if len(items) >= 2 or any(item.confidence >= .95 for item in items):
            verified.extend(items)
    return sorted(verified, key=lambda item: item.timestamp_sec)


_MIRAGE_POSITION_LABELS = {
    "palaceinterior": "palace_interior",
    "bombsitea": "bombsite_a",
    "bombsiteb": "bombsite_b",
    "tstart": "t_start",
    "apartments": "apartments",
    "tramp": "t_ramp",
    "underpass": "underpass",
    "ctstart": "ct_start",
    "shop": "shop",
    "middle": "middle",
    "snipersnest": "snipers_nest",
    "sidealley": "side_alley",
    "backalley": "back_alley",
    "palacealley": "palace_alley",
    "house": "house",
    "stairs": "stairs",
    "connector": "connector",
    "jungle": "jungle",
    "catwalk": "catwalk",
    "topofmid": "top_mid",
}
_MIRAGE_STRONG_ANCHORS = {"palace_interior", "snipers_nest"}
_MIRAGE_SUPPORTING_ANCHORS = {
    "underpass", "apartments", "t_ramp", "shop", "side_alley",
}


def _normalize_position_observation(
    observation: ContextObservation,
) -> ContextObservation | None:
    raw = "".join(ch.casefold() for ch in str(observation.value) if ch.isalpha())
    if len(raw) < 3 or observation.confidence < .30:
        return None
    matches: list[tuple[float, str]] = []
    for label, canonical in _MIRAGE_POSITION_LABELS.items():
        ratio = SequenceMatcher(None, raw, label).ratio()
        matches.append((ratio, canonical))
    matches.sort(reverse=True)
    best_ratio, best_label = matches[0]
    if best_ratio < .72:
        return None
    # A damaged final A/B glyph can make "Bombsite E" equally similar to both
    # sites.  Tied canonical matches are rejected instead of choosing by dict
    # order and fabricating a site.
    if (
        len(matches) > 1
        and matches[1][1] != best_label
        and best_ratio - matches[1][0] < .03
    ):
        return None
    return ContextObservation(
        frame_index=observation.frame_index,
        timestamp_sec=observation.timestamp_sec,
        value=best_label,
        confidence=min(1.0, observation.confidence * best_ratio),
        source=observation.source,
    )


def _verified_position_observations(
    observations: Sequence[ContextObservation],
) -> list[ContextObservation]:
    by_position: dict[str, list[ContextObservation]] = {}
    for item in observations:
        by_position.setdefault(str(item.value), []).append(item)
    verified: list[ContextObservation] = []
    for items in by_position.values():
        if len(items) >= 2 or any(item.confidence >= .60 for item in items):
            verified.extend(items)
    return sorted(verified, key=lambda item: item.timestamp_sec)


def _resolve_map(
    observations: Sequence[ContextObservation],
) -> tuple[str | None, float]:
    confident = {
        str(item.value) for item in observations if item.confidence >= .60
    }
    if confident & _MIRAGE_STRONG_ANCHORS:
        return "mirage", .95
    supporting = confident & _MIRAGE_SUPPORTING_ANCHORS
    if len(supporting) >= 2:
        return "mirage", .85
    return None, 0.0
