"""First-person visual signals that do not depend on player identity.

Only the gameplay viewport and native screen effects are measured.  Bottom
overlays, creator watermarks, names, chat, and kill-feed text are excluded.
The native kill-feed's local-player highlight frame is measured geometrically;
no name or other feed text is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from gamesight.domain.models import EventType, Evidence, GameEvent, RoundAnalysis
from gamesight.perception.weapon import (
    FirstPersonUtilityClassifier, FirstPersonWeaponClassifier,
)


@dataclass(frozen=True)
class FirstPersonSample:
    frame_index: int
    timestamp_sec: float
    flashed: bool
    scoped: bool
    motion_score: float | None
    player_team: str | None = None
    shot_candidate: bool = False
    damage_candidate: bool = False
    shot_signal_score: float = 0.0
    damage_signal_score: float = 0.0
    health_hud_visible: bool = False
    health_hud_score: float = 0.0
    # Native top scoreboard cards provide an independent POV life-state
    # signal.  Both candidate sides are retained because teams swap sides at
    # halftime; round aggregation selects the card with the stable orange POV
    # highlight instead of assuming a fixed screen side.
    player_card_left_alive: bool | None = None
    player_card_right_alive: bool | None = None
    player_card_left_selected_score: float = 0.0
    player_card_right_selected_score: float = 0.0
    local_kill_highlight: bool = False
    local_kill_highlight_score: float = 0.0
    # Encoded crops of native red-highlighted kill-feed rows.  They are used
    # only to recognise a row that persists or moves in the feed; names are
    # never interpreted for player identity or exposed in the report.
    local_kill_row_fingerprints: tuple[bytes, ...] = ()
    # Vertical feed positions aligned with ``local_kill_row_fingerprints``.
    # They let the temporal tracker keep a fading/compressed row attached to
    # its original kill even when its glyph fingerprint changes noticeably.
    local_kill_row_positions: tuple[float, ...] = ()
    weapon: str | None = None
    weapon_confidence: float = 0.0
    weapon_source: str | None = None
    held_utility: str | None = None
    utility_confidence: float = 0.0
    utility_source: str | None = None
    # Diagnostic only: how strongly the raw frame delta was concentrated in
    # a few tiles.  Creator overlays (keystrokes, voice widgets, subtitles)
    # usually change locally, while camera motion changes most of the view.
    localized_overlay_score: float = 0.0


@dataclass(frozen=True)
class NormalizedOverlayRegion:
    """A non-gameplay overlay excluded from *motion only*.

    Coordinates are normalized to the full video frame.  These regions never
    suppress native HUD, kill-feed, weapon, or death evidence.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = "creator_overlay"

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.x_min < self.x_max <= 1.0
            and 0.0 <= self.y_min < self.y_max <= 1.0
        ):
            raise ValueError("overlay region coordinates must be within 0..1")


# A common streamer input display occupies the lower centre.  It is outside
# the crosshair/engagement area but overlaps the old generic motion crop.
# Excluding it only from motion is safe for recordings without the widget too:
# no combat/HUD detector consumes this mask.
DEFAULT_MOTION_OVERLAY_REGIONS: tuple[NormalizedOverlayRegion, ...] = (
    NormalizedOverlayRegion(
        x_min=.35, y_min=.70, x_max=.64, y_max=.91,
        label="center_bottom_input_display",
    ),
)


class FirstPersonAnalyzer:
    """Measure flash exposure, scope state, and camera motion per frame."""

    def __init__(
        self, weapon_classifier: FirstPersonWeaponClassifier | None = None,
        utility_classifier: FirstPersonUtilityClassifier | None = None,
        motion_overlay_regions: Sequence[NormalizedOverlayRegion] | None = None,
    ) -> None:
        self._previous: NDArray[np.uint8] | None = None
        self._previous_ts: float | None = None
        self._player_team: str | None = None
        self._previous_muzzle_mask: NDArray[np.bool_] | None = None
        self._red_edge_baseline: tuple[float, float] | None = None
        self._weapon_classifier = weapon_classifier or FirstPersonWeaponClassifier()
        self._utility_classifier = utility_classifier or FirstPersonUtilityClassifier()
        self._motion_overlay_regions = tuple(
            DEFAULT_MOTION_OVERLAY_REGIONS
            if motion_overlay_regions is None else motion_overlay_regions
        )

    def update(
        self, image: NDArray[np.uint8], frame_index: int, timestamp_sec: float
    ) -> FirstPersonSample:
        # Full-HD colour conversion on every 10 FPS sample is unnecessary for
        # screen-wide flash/scope geometry.  A 640px working frame preserves
        # the signal while cutting per-frame pixel work by roughly 9x at 1080p.
        source_h, source_w = image.shape[:2]
        if source_w > 640:
            working = cv2.resize(
                image,
                (640, max(1, int(round(source_h * 640 / source_w)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        h, w = gray.shape[:2]

        mean = float(gray.mean())
        std = float(gray.std())
        flashed = mean > 205.0 and std < 65.0

        # Four off-centre patches sit outside the circular AWP viewport.  All
        # become black while scoped; requiring all four avoids dark-map noise.
        patches = (
            gray[int(h * .18):int(h * .38), int(w * .08):int(w * .27)],
            gray[int(h * .18):int(h * .38), int(w * .73):int(w * .92)],
            gray[int(h * .62):int(h * .82), int(w * .08):int(w * .27)],
            gray[int(h * .62):int(h * .82), int(w * .73):int(w * .92)],
        )
        dark_ratio = float(np.mean([np.mean(patch < 18) for patch in patches]))
        scoped = dark_ratio > 0.72 and mean < 105.0

        # Central gameplay viewport explicitly excludes all HUD edges and the
        # bottom watermark band.  A validity mask removes known creator UI,
        # while robust tile aggregation rejects local overlay changes at any
        # other position (voice widgets, chat popups, subtitles).  Camera
        # movement affects most tiles; a UI animation normally affects a few.
        viewport = gray[int(h * .16):int(h * .84), int(w * .10):int(w * .90)]
        small = cv2.resize(viewport, (160, 90), interpolation=cv2.INTER_AREA)
        motion_mask = _motion_validity_mask(
            full_height=h,
            full_width=w,
            viewport_shape=small.shape,
            regions=self._motion_overlay_regions,
        )
        motion: float | None = None
        localized_overlay_score = 0.0
        if self._previous is not None and self._previous_ts is not None:
            delta_sec = max(timestamp_sec - self._previous_ts, 0.05)
            difference, localized_overlay_score = _robust_frame_difference(
                small, self._previous, motion_mask,
            )
            motion = min(1.0, difference / delta_sec)
        self._previous = small
        self._previous_ts = timestamp_sec

        # The native CS2 team emblem sits at bottom-centre.  Restricting this
        # classifier to that HUD element avoids creator watermarks and names.
        detected_team = _detect_player_team(working)
        if detected_team is not None:
            self._player_team = detected_team

        shot_score, damage_score = self._combat_signal_scores(hsv)
        health_hud_score, health_hud_visible = _native_health_hud(working)
        (
            player_card_left_alive,
            player_card_right_alive,
            player_card_left_score,
            player_card_right_score,
        ) = _native_player_card_signals(working)
        local_kill_score = _native_local_kill_highlight(working)
        local_kill_rows = (
            _native_local_kill_rows(image)
            if local_kill_score >= .12 else ()
        )
        local_kill_highlight = local_kill_score >= .45 or bool(local_kill_rows)
        if local_kill_rows:
            local_kill_score = max(local_kill_score, .55)
        # These are deliberately candidates, not claims of a confirmed shot
        # or hit.  They only upgrade an encounter when an opposing character
        # is detected in the same short time window.
        shot_candidate = not flashed and shot_score >= 0.04
        damage_candidate = not flashed and damage_score >= 0.025
        # Do not classify death/spectator/end screens.  A native first-person
        # health cluster must be visible in the same frame as the held item.
        weapon = (
            self._weapon_classifier.classify(working, scoped=scoped)
            if health_hud_visible else None
        )
        utility = (
            self._utility_classifier.classify(working)
            if health_hud_visible else None
        )

        return FirstPersonSample(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            flashed=flashed,
            scoped=scoped,
            motion_score=motion,
            player_team=self._player_team,
            shot_candidate=shot_candidate,
            damage_candidate=damage_candidate,
            shot_signal_score=round(shot_score, 5),
            damage_signal_score=round(damage_score, 5),
            health_hud_visible=health_hud_visible,
            health_hud_score=round(health_hud_score, 5),
            player_card_left_alive=player_card_left_alive,
            player_card_right_alive=player_card_right_alive,
            player_card_left_selected_score=round(player_card_left_score, 5),
            player_card_right_selected_score=round(player_card_right_score, 5),
            local_kill_highlight=local_kill_highlight,
            local_kill_highlight_score=round(local_kill_score, 5),
            local_kill_row_fingerprints=tuple(row[0] for row in local_kill_rows),
            local_kill_row_positions=tuple(row[1] for row in local_kill_rows),
            weapon=weapon.category if weapon is not None else None,
            weapon_confidence=(weapon.confidence if weapon is not None else 0.0),
            weapon_source=(weapon.source if weapon is not None else None),
            held_utility=(utility.category if utility is not None else None),
            utility_confidence=(utility.confidence if utility is not None else 0.0),
            utility_source=(utility.source if utility is not None else None),
            localized_overlay_score=round(localized_overlay_score, 5),
        )

    def _combat_signal_scores(
        self, hsv: NDArray[np.uint8],
    ) -> tuple[float, float]:
        """Return conservative muzzle-flash and damage-overlay change scores."""
        h, w = hsv.shape[:2]
        muzzle = hsv[int(h * .30):int(h * .78), int(w * .30):int(w * .86)]
        warm = (
            (muzzle[:, :, 0] <= 34)
            & (muzzle[:, :, 1] >= 110)
            & (muzzle[:, :, 2] >= 215)
        )
        shot_score = 0.0
        if (
            self._previous_muzzle_mask is not None
            and self._previous_muzzle_mask.shape == warm.shape
        ):
            shot_score = float(np.mean(warm & ~self._previous_muzzle_mask))
        self._previous_muzzle_mask = warm

        red = (
            ((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 172))
            & (hsv[:, :, 1] >= 120)
            & (hsv[:, :, 2] >= 100)
        )
        left = float(np.mean(
            red[int(h * .12):int(h * .82), int(w * .03):int(w * .20)]
        ))
        right = float(np.mean(
            red[int(h * .12):int(h * .82), int(w * .80):int(w * .97)]
        ))
        damage_score = 0.0
        if self._red_edge_baseline is not None:
            damage_score = min(
                max(0.0, left - self._red_edge_baseline[0]),
                max(0.0, right - self._red_edge_baseline[1]),
            )
        if self._red_edge_baseline is None:
            self._red_edge_baseline = (left, right)
        else:
            self._red_edge_baseline = (
                self._red_edge_baseline[0] * .85 + left * .15,
                self._red_edge_baseline[1] * .85 + right * .15,
            )
        # Require visible red on both sides; a single red wall or weapon skin
        # must not be interpreted as player damage.
        if min(left, right) < .035:
            damage_score = 0.0
        return shot_score, damage_score


def _native_local_kill_highlight(image: NDArray[np.uint8]) -> float:
    """Measure CS2's red local-kill frame without reading kill-feed text.

    Native CS2 draws a thin red/magenta rectangle around a feed row when the
    POV player is the killer.  Other players' rows retain only the dark feed
    background.  We require both a long horizontal chroma segment and an
    adjacent dark row interior so warm map surfaces do not become kills.
    """
    h, w = image.shape[:2]
    roi = image[
        int(h * .003):int(h * .253),
        int(w * .785):int(w * .997),
    ]
    if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 24:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (
        ((hue <= 12) | (hue >= 150))
        & (saturation >= 140)
        & (value >= 70)
    ).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        red,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(17, roi.shape[1] // 8), 1),
        ),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal)
    best = 0.0
    vertical_limit = roi.shape[0] * .08
    minimum_width = roi.shape[1] * .20
    pad = max(2, roi.shape[0] // 60)
    depth = max(12, roi.shape[0] // 6)
    for x, y, component_w, component_h, area in stats[1:]:
        if component_w < minimum_width or component_h > vertical_limit:
            continue
        # A true HUD outline is a nearly continuous thin line.  Warm walls,
        # wood grain, and damage tint produce thicker, broken colour blobs.
        fill_ratio = area / max(1, component_w * component_h)
        if fill_ratio < .70:
            continue
        component_pixels = horizontal[
            y:y + component_h, x:x + component_w
        ] > 0
        component_value = value[
            y:y + component_h, x:x + component_w
        ][component_pixels]
        if component_value.size == 0 or float(np.median(component_value)) < 95:
            continue
        adjacent_row_scores: list[tuple[float, float]] = []
        for y1, y2 in (
            (y + pad, min(y + depth, roi.shape[0])),
            (max(0, y - depth), max(0, y - pad)),
        ):
            if y2 > y1:
                row_saturation = saturation[y1:y2, x:x + component_w]
                row_value = value[y1:y2, x:x + component_w]
                adjacent_row_scores.append((
                    float(np.mean(row_value < 110)),
                    float(np.mean(
                        (row_value >= 170) & (row_saturation <= 105)
                    )),
                ))
        qualified_rows = [
            dark for dark, white in adjacent_row_scores
            if dark >= .20 and white >= .01
        ]
        if not qualified_rows:
            continue
        dark_ratio = max(qualified_rows)
        width_ratio = component_w / roi.shape[1]
        best = max(best, width_ratio * min(1.0, dark_ratio / .50))
    return min(1.0, best)


def _native_local_kill_row_fingerprints(
    image: NDArray[np.uint8],
) -> tuple[bytes, ...]:
    """Compatibility wrapper returning only visual row fingerprints."""
    return tuple(row[0] for row in _native_local_kill_rows(image))


def _native_local_kill_rows(
    image: NDArray[np.uint8],
) -> tuple[tuple[bytes, float], ...]:
    """Return compact visual crops for native local-kill rows.

    The crop is deliberately content-agnostic.  It is never OCRed and does
    not decide who made the kill; the native red frame already made that
    decision.  Keeping the glyph layout lets the aggregator recognise the
    same feed row after it moves upward or briefly fades out.
    """
    h, w = image.shape[:2]
    y0, y1 = int(h * .003), int(h * .253)
    x0, x1 = int(w * .70), int(w * .997)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 60:
        return ()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (
        ((hue <= 12) | (hue >= 150))
        & (saturation >= 120)
        & (value >= 60)
    ).astype(np.uint8) * 255
    scale = max(1, round(roi.shape[1] / 570))
    candidates: list[tuple[int, int, int, int]] = []
    minimum_width = roi.shape[1] * .22
    minimum_height = max(8, roi.shape[0] * .055)
    maximum_height = roi.shape[0] * .24

    def add_candidate(candidate: tuple[int, int, int, int]) -> None:
        x, y, component_w, component_h = candidate
        if (
            component_w < minimum_width
            or not minimum_height <= component_h <= maximum_height
            # Native kill-feed rows are right-aligned against the screen
            # edge.  Requiring that geometry prevents red map details or a
            # plain teammate row below the local highlight from becoming a
            # second fingerprint.
            or x + component_w < roi.shape[1] * .90
        ):
            return
        inset = max(2, component_h // 10)
        interior = roi[
            y + inset:y + component_h - inset,
            x + inset:x + component_w - inset,
        ]
        if interior.size == 0:
            return
        interior_hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
        interior_saturation = interior_hsv[:, :, 1]
        interior_value = interior_hsv[:, :, 2]
        dark_ratio = float(np.mean(interior_value < 115))
        glyph_ratio = float(np.mean(
            (interior_value >= 155) & (interior_saturation <= 150)
        ))
        if dark_ratio >= .18 and glyph_ratio >= .008:
            candidates.append(candidate)

    # Adjacent local kills share a border and therefore become one large
    # external contour.  Pair horizontal outline segments first so two stacked
    # rows remain two distinct observations.  Compression can break the thin
    # top edge into nearby fragments, so reconnect only very small horizontal
    # gaps before looking for long lines.
    joined_red = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5 * scale, 1)),
    )
    horizontal = cv2.morphologyEx(
        joined_red,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(17, round(roi.shape[1] * .075)), 1),
        ),
    )
    _, _, line_stats, _ = cv2.connectedComponentsWithStats(horizontal)
    raw_lines: list[tuple[int, int, int]] = []
    for x, y, line_w, line_h, area in line_stats[1:]:
        if (
            line_w >= minimum_width
            and line_h <= max(4, roi.shape[0] * .055)
            and area / max(1, line_w * line_h) >= .52
        ):
            raw_lines.append((x, y + line_h // 2, line_w))
    raw_lines.sort(key=lambda item: item[1])
    line_groups: list[list[tuple[int, int, int]]] = []
    for line in raw_lines:
        if (
            not line_groups
            or line[1] - line_groups[-1][-1][1] > max(4, roi.shape[0] * .025)
        ):
            line_groups.append([line])
        else:
            line_groups[-1].append(line)
    lines = [
        (
            min(item[0] for item in group),
            round(float(np.median([item[1] for item in group]))),
            max(item[0] + item[2] for item in group),
        )
        for group in line_groups
    ]
    for upper, lower in zip(lines, lines[1:]):
        gap = lower[1] - upper[1]
        overlap = min(upper[2], lower[2]) - max(upper[0], lower[0])
        if (
            minimum_height <= gap <= maximum_height
            and overlap >= minimum_width * .55
        ):
            x = min(upper[0], lower[0])
            right = max(upper[2], lower[2])
            add_candidate((x, upper[1], right - x, gap))

    # A single fading row may not retain both horizontal edges.  The closed
    # outline contour is the fallback for that case.
    closed = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9 * scale, 5 * scale)),
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        x, y, component_w, component_h = cv2.boundingRect(contour)
        add_candidate((x, y, component_w, component_h))

    # Contours from a fading outline can overlap.  Keep the wider rectangle
    # when two candidates describe the same vertical feed row.
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for candidate in candidates:
        _, y, _, component_h = candidate
        centre = y + component_h / 2
        if any(
            abs(centre - (other_y + other_h / 2))
            < max(component_h, other_h) * .55
            for _, other_y, _, other_h in selected
        ):
            continue
        selected.append(candidate)

    encoded: list[tuple[bytes, float]] = []
    for x, y, component_w, component_h in sorted(
        selected, key=lambda item: item[1],
    ):
        inset = max(2, component_h // 10)
        row = roi[
            y + inset:y + component_h - inset,
            x + round(component_w * .65):x + component_w - inset,
        ]
        if row.size == 0:
            continue
        normalised = cv2.resize(
            row, (200, 40), interpolation=cv2.INTER_AREA,
        )
        ok, buffer = cv2.imencode(
            ".jpg", normalised, [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if ok:
            centre_y = (y + component_h / 2) / roi.shape[0]
            encoded.append((buffer.tobytes(), round(centre_y, 5)))
    return tuple(encoded)


def _motion_validity_mask(
    *,
    full_height: int,
    full_width: int,
    viewport_shape: tuple[int, ...],
    regions: Sequence[NormalizedOverlayRegion],
) -> NDArray[np.bool_]:
    """Map normalized full-frame exclusions into the motion thumbnail."""
    del full_height, full_width  # coordinates are normalized by design
    height, width = viewport_shape[:2]
    mask = np.ones((height, width), dtype=np.bool_)
    # Must match the gameplay viewport in ``update`` above.
    viewport_x_min, viewport_x_max = .10, .90
    viewport_y_min, viewport_y_max = .16, .84
    for region in regions:
        x_min = max(region.x_min, viewport_x_min)
        x_max = min(region.x_max, viewport_x_max)
        y_min = max(region.y_min, viewport_y_min)
        y_max = min(region.y_max, viewport_y_max)
        if x_min >= x_max or y_min >= y_max:
            continue
        left = int(np.floor(
            (x_min - viewport_x_min) / (viewport_x_max - viewport_x_min) * width
        ))
        right = int(np.ceil(
            (x_max - viewport_x_min) / (viewport_x_max - viewport_x_min) * width
        ))
        top = int(np.floor(
            (y_min - viewport_y_min) / (viewport_y_max - viewport_y_min) * height
        ))
        bottom = int(np.ceil(
            (y_max - viewport_y_min) / (viewport_y_max - viewport_y_min) * height
        ))
        mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = False
    return mask


def _robust_frame_difference(
    current: NDArray[np.uint8],
    previous: NDArray[np.uint8],
    validity_mask: NDArray[np.bool_],
    *,
    tile_rows: int = 6,
    tile_columns: int = 8,
) -> tuple[float, float]:
    """Return de-noised motion and a local-overlay diagnostic score.

    The top 20% most-changing tiles are omitted from the motion value.  This
    retains scene-wide camera movement but prevents a fixed keyboard display,
    relocated voice UI, animated watermark, or subtitle from dominating the
    result.  ``localized_score`` is zero for uniform change and approaches one
    when nearly all raw change is concentrated in those few tiles.
    """
    if current.shape != previous.shape or current.shape != validity_mask.shape:
        return 0.0, 0.0
    delta = cv2.absdiff(current, previous).astype(np.float32) / 255.0
    height, width = delta.shape[:2]
    tile_values: list[float] = []
    for row in range(tile_rows):
        top = row * height // tile_rows
        bottom = (row + 1) * height // tile_rows
        for column in range(tile_columns):
            left = column * width // tile_columns
            right = (column + 1) * width // tile_columns
            tile_mask = validity_mask[top:bottom, left:right]
            # Mostly masked tiles contain no trustworthy gameplay pixels.
            if tile_mask.size == 0 or float(np.mean(tile_mask)) < .50:
                continue
            values = delta[top:bottom, left:right][tile_mask]
            if values.size:
                tile_values.append(float(np.mean(values)))
    if not tile_values:
        return 0.0, 0.0
    ordered = np.sort(np.asarray(tile_values, dtype=np.float32))
    trim_count = max(1, int(np.ceil(len(ordered) * .20)))
    retained = ordered[:-trim_count] if len(ordered) > trim_count else ordered
    robust_difference = float(np.mean(retained))
    total = float(np.sum(ordered))
    if total <= 1e-9:
        return robust_difference, 0.0
    top_share = float(np.sum(ordered[-trim_count:]) / total)
    expected_share = trim_count / len(ordered)
    localized_score = max(
        0.0,
        min(1.0, (top_share - expected_share) / max(1.0 - expected_share, 1e-6)),
    )
    return robust_difference, localized_score


def _detect_player_team(image: NDArray[np.uint8]) -> str | None:
    """Infer the POV side from the native bottom-centre team-colour HUD."""
    h, w = image.shape[:2]
    roi = image[int(h * .86):int(h * .995), int(w * .43):int(w * .57)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    terrorist = int(np.count_nonzero(cv2.inRange(
        hsv, np.array((10, 80, 80)), np.array((45, 255, 255)),
    )))
    counter_terrorist = int(np.count_nonzero(cv2.inRange(
        hsv, np.array((75, 70, 70)), np.array((130, 255, 255)),
    )))
    minimum = max(12, int(roi.shape[0] * roi.shape[1] * .001))
    if terrorist >= minimum and terrorist > counter_terrorist * 1.8:
        return "t"
    if counter_terrorist >= minimum and counter_terrorist > terrorist * 1.8:
        return "ct"
    return None


def _native_health_hud(image: NDArray[np.uint8]) -> tuple[float, bool]:
    """Measure the native bottom HUD health/armour cluster.

    This intentionally does not try to turn coloured-pixel area into an HP
    number.  A stable CS2 HUD cluster is sufficient for conservative death
    transition detection, while arbitrary recording watermarks are outside
    this narrow native-HUD region.
    """
    h, w = image.shape[:2]
    roi = image[int(h * .90):int(h * .995), int(w * .245):int(w * .335)]
    if roi.size == 0:
        return 0.0, False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    coloured_text = float(np.mean(
        (hsv[:, :, 2] > 145) & (hsv[:, :, 1] > 55)
    ))
    # Some HUD themes/side colours render the HP digits almost white.  The
    # old colour-only gate therefore flickered off for one or two samples even
    # while a clear "100" remained visible.  Accept low-saturation glyphs only
    # when their edge density is high enough to look like text; a bright wall
    # or death fade has area but not this repeated digit structure.
    white_mask = (
        (hsv[:, :, 2] > 160) & (hsv[:, :, 1] < 70)
    )
    white_text = float(np.mean(white_mask))
    edge_ratio = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
    coloured_visible = coloured_text >= .025 and edge_ratio >= .02
    white_components, _labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(white_mask.astype(np.uint8))
    )
    glyph_count = 0
    for index in range(1, white_components):
        _x, _y, component_width, component_height, area = stats[index]
        aspect = component_width / max(component_height, 1)
        if (
            3 <= component_width <= 12
            and 6 <= component_height <= 18
            and 15 <= area <= 110
            and .30 <= aspect <= 1.50
        ):
            glyph_count += 1
    white_visible = (
        white_text >= .045 and edge_ratio >= .08 and glyph_count >= 2
    )
    score = coloured_text + white_text * .65 + edge_ratio * .35
    return score, coloured_visible or white_visible


def _native_player_card_signals(
    image: NDArray[np.uint8],
) -> tuple[bool | None, bool | None, float, float]:
    """Read the two possible native POV cards at the top scoreboard.

    FACEIT/demo recordings move the POV card from the right roster to the
    left roster after the side swap.  The selected card has an orange native
    background; its HP glyph/bar area has dense edges and variance while the
    dead card becomes a flat dim panel.  Only these fixed native HUD boxes are
    measured, so creator voice/keyboard overlays cannot affect the result.
    """
    h, w = image.shape[:2]
    if h < 180 or w < 320:
        return None, None, 0.0, 0.0

    def card(x_min: float, x_max: float) -> tuple[bool | None, float]:
        panel = image[0:int(h * .18), int(w * x_min):int(w * x_max)]
        hp = image[
            int(h * .052):int(h * .087),
            int(w * x_min):int(w * x_max),
        ]
        if panel.size == 0 or hp.size == 0 or hp.shape[0] < 8:
            return None, 0.0
        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        selected_score = float(np.mean(
            (hsv[:, :, 0] >= 5)
            & (hsv[:, :, 0] <= 35)
            & (hsv[:, :, 1] >= 65)
            & (hsv[:, :, 2] >= 55)
        ))
        gray = cv2.cvtColor(hp, cv2.COLOR_BGR2GRAY)
        edge_ratio = float(np.mean(cv2.Canny(gray, 60, 130) > 0))
        contrast = float(gray.std())
        alive = contrast >= 34.0 and edge_ratio >= .20
        return alive, selected_score

    left_alive, left_score = card(.424, .471)
    right_alive, right_score = card(.691, .739)
    return left_alive, right_alive, left_score, right_score


def build_first_person_summary_events(
    rounds: list[RoundAnalysis], samples: list[FirstPersonSample]
) -> list[GameEvent]:
    """Build a neutral round summary plus timestamped high-confidence moments.

    Raw scene motion is intentionally descriptive.  Running, turning, weapon
    switching, and actual aim corrections all move the viewport, so motion by
    itself must never be presented as an aim error or an engagement.
    """
    events: list[GameEvent] = []
    for round_analysis in rounds:
        if round_analysis.end_sec is None:
            continue
        current = [
            sample for sample in samples
            if round_analysis.start_sec <= sample.timestamp_sec < round_analysis.end_sec
        ]
        if not current:
            continue

        sample_durations = _sample_durations(
            current, round_analysis.end_sec,
        )
        duration_by_frame = {
            sample.frame_index: sample_duration
            for sample, sample_duration in zip(current, sample_durations)
        }
        flash_sec = round(sum(
            sample_duration for sample, sample_duration
            in zip(current, sample_durations) if sample.flashed
        ), 2)
        scoped_sec = round(sum(
            sample_duration for sample, sample_duration
            in zip(current, sample_durations) if sample.scoped
        ), 2)
        duration = max(round_analysis.end_sec - round_analysis.start_sec, 0.001)

        valid_motion = [
            sample for sample in current
            if sample.motion_score is not None and not sample.flashed
        ]
        motion_weight = sum(
            duration_by_frame[sample.frame_index] for sample in valid_motion
        )
        motion_avg = (
            sum(
                float(sample.motion_score or 0.0)
                * duration_by_frame[sample.frame_index]
                for sample in valid_motion
            ) / motion_weight if motion_weight else 0.0
        )
        stationary_ratio = (
            sum(
                duration_by_frame[sample.frame_index]
                for sample in valid_motion if sample.motion_score < 0.12
            ) / motion_weight if motion_weight else 0.0
        )

        flash_count = 0
        previously_flashed = False
        for sample in current:
            if sample.flashed and not previously_flashed:
                flash_count += 1
            previously_flashed = sample.flashed

        # A summary frame should represent the playable part of the round,
        # never simply the maximum-motion opening frame.  Use the first
        # high-confidence visual moment, otherwise a mid-round frame after
        # CS2's typical 15-second opening traversal window.
        moment_samples = [sample for sample in current if sample.flashed or sample.scoped]
        target_sec = round_analysis.start_sec + max(15.0, duration * 0.55)
        notable = moment_samples[0] if moment_samples else min(
            current, key=lambda sample: abs(sample.timestamp_sec - target_sec),
        )

        index = len(events) + 1
        events.append(GameEvent(
            event_id=f"first_person_summary_{index:03d}",
            event_type=EventType.FIRST_PERSON_SUMMARY,
            start_sec=notable.timestamp_sec,
            confidence=0.88,
            evidence=[Evidence(
                frame_index=notable.frame_index,
                timestamp_sec=notable.timestamp_sec,
                source="FirstPersonAnalyzer.gameplay_viewport",
            )],
            attributes={
                "round_id": round_analysis.round_id,
                "flash_count": flash_count,
                "flash_exposure_sec": flash_sec,
                "scoped_sec": scoped_sec,
                "scoped_ratio": round(scoped_sec / duration, 4),
                "view_motion_avg": round(motion_avg, 4),
                "stationary_ratio": round(stationary_ratio, 4),
                "motion_is_descriptive": True,
            },
        ))

        # Preserve individual flash/scope episodes so one round can yield
        # several reviewable moments instead of a single aggregate card.
        episodes = [
            ("flash", episode) for episode in _episodes(current, "flashed")
        ] + [
            ("scope", episode) for episode in _episodes(current, "scoped")
        ]
        episodes.sort(key=lambda item: item[1][0].timestamp_sec)
        for moment_index, (kind, episode) in enumerate(episodes[:4], start=1):
            episode_duration = sum(
                duration_by_frame[sample.frame_index] for sample in episode
            )
            # Ignore single noisy samples while retaining meaningful flashes;
            # scope advice needs a longer, continuous hold.
            minimum = 1.0 if kind == "flash" else 4.0
            if episode_duration < minimum:
                continue
            first = episode[0]
            events.append(GameEvent(
                event_id=(
                    f"first_person_{kind}_{round_analysis.round_id}_{moment_index:02d}"
                ),
                event_type=EventType.FIRST_PERSON_MOMENT,
                start_sec=first.timestamp_sec,
                end_sec=round(first.timestamp_sec + episode_duration, 2),
                confidence=0.88 if kind == "flash" else 0.86,
                evidence=[Evidence(
                    frame_index=first.frame_index,
                    timestamp_sec=first.timestamp_sec,
                    source=f"FirstPersonAnalyzer.{kind}_episode",
                )],
                attributes={
                    "round_id": round_analysis.round_id,
                    "moment_kind": kind,
                    "duration_sec": round(episode_duration, 2),
                },
            ))
    return events


def _sample_durations(
    samples: list[FirstPersonSample], round_end_sec: float,
) -> list[float]:
    """Return time weights that remain correct for mixed 2/10 FPS samples."""
    if not samples:
        return []
    durations = [
        max(0.0, later.timestamp_sec - sample.timestamp_sec)
        for sample, later in zip(samples, samples[1:])
    ]
    durations.append(max(0.0, round_end_sec - samples[-1].timestamp_sec))
    return durations


def _episodes(
    samples: list[FirstPersonSample], attribute: str
) -> list[list[FirstPersonSample]]:
    """Group consecutive true samples into visual-effect episodes."""
    result: list[list[FirstPersonSample]] = []
    active: list[FirstPersonSample] = []
    for sample in samples:
        if bool(getattr(sample, attribute)):
            active.append(sample)
        elif active:
            result.append(active)
            active = []
    if active:
        result.append(active)
    return result
