"""Conservative first-person held-weapon category evidence.

The POV view is the primary source.  An optional native equipment-panel hint
may support, but never replace, the held-item observation.  Categories that
cannot be established from a high-specificity visual signal remain unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


WEAPON_CATEGORIES = frozenset({
    "pistol", "rifle", "smg", "shotgun", "sniper", "knife", "grenade", "c4",
})
UTILITY_CATEGORIES = frozenset({
    "flashbang", "smoke", "he_grenade", "molotov", "incendiary", "decoy",
})


@dataclass(frozen=True)
class WeaponObservation:
    category: str
    confidence: float
    source: str


class HeldWeaponBackend(Protocol):
    """Interface for a future trained held-weapon image classifier."""

    def classify(self, image: NDArray[np.uint8]) -> WeaponObservation | None: ...


class HeldUtilityBackend(Protocol):
    """Interface for a trained first-person held-utility classifier."""

    def classify(self, image: NDArray[np.uint8]) -> WeaponObservation | None: ...


class FirstPersonUtilityClassifier:
    """Accept only a trained held-view result; inventory hints are auxiliary."""

    def __init__(self, backend: HeldUtilityBackend | None = None) -> None:
        self._backend = backend

    def classify(
        self,
        image: NDArray[np.uint8],
        *,
        equipment_hint: WeaponObservation | None = None,
    ) -> WeaponObservation | None:
        if self._backend is None:
            return None
        primary = self._backend.classify(image)
        if (
            primary is None
            or primary.category not in UTILITY_CATEGORIES
            or primary.confidence < .85
        ):
            return None
        if (
            equipment_hint is not None
            and equipment_hint.category == primary.category
            and equipment_hint.category in UTILITY_CATEGORIES
        ):
            return WeaponObservation(
                category=primary.category,
                confidence=min(
                    1.0,
                    max(primary.confidence, equipment_hint.confidence) + .02,
                ),
                source=f"{primary.source}+{equipment_hint.source}",
            )
        return primary


class FirstPersonWeaponClassifier:
    """Classify only high-specificity held-item signals from the POV image."""

    def __init__(self, backend: HeldWeaponBackend | None = None) -> None:
        self._backend = backend

    def classify(
        self,
        image: NDArray[np.uint8],
        *,
        scoped: bool = False,
        equipment_hint: WeaponObservation | None = None,
    ) -> WeaponObservation | None:
        primary: WeaponObservation | None = None
        if scoped:
            # CS2's full circular scope view is a direct first-person sniper
            # signal; this is not inferred from ammo, the kill feed, or text.
            primary = WeaponObservation(
                category="sniper", confidence=.97,
                source="FirstPersonWeaponClassifier.native_scope_view",
            )
        elif _looks_like_held_c4(image):
            primary = WeaponObservation(
                category="c4", confidence=.96,
                source="FirstPersonWeaponClassifier.held_c4_geometry",
            )
        elif self._backend is not None:
            candidate = self._backend.classify(image)
            if (
                candidate is not None
                and candidate.category in WEAPON_CATEGORIES
                and candidate.confidence >= .80
            ):
                primary = candidate

        # Bottom-right equipment highlighting is auxiliary only.  It can raise
        # confidence when it agrees with the held view, but absence cannot
        # suppress a valid first-person observation and disagreement is not
        # resolved by guessing.
        if primary is None:
            return None
        if (
            equipment_hint is not None
            and equipment_hint.category == primary.category
            and equipment_hint.category in WEAPON_CATEGORIES
        ):
            return WeaponObservation(
                category=primary.category,
                confidence=min(1.0, max(primary.confidence, equipment_hint.confidence) + .02),
                source=f"{primary.source}+{equipment_hint.source}",
            )
        return primary


def _looks_like_held_c4(image: NDArray[np.uint8]) -> bool:
    """Detect the distinctive central held C4 display/keypad conservatively."""
    if image.size == 0:
        return False
    h, w = image.shape[:2]
    # The CS2 held C4 is centred slightly right and reaches the bottom edge.
    roi = image[int(h * .40):int(h * .995), int(w * .32):int(w * .82)]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= 34) & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 70)
    ).astype(np.uint8) * 255
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(green)
    roi_area = float(roi.shape[0] * roi.shape[1])
    display_found = False
    for index in range(1, count):
        x, y, cw, ch, area = stats[index]
        aspect = cw / max(ch, 1)
        fraction = area / max(roi_area, 1.0)
        centre_x = (x + cw / 2) / roi.shape[1]
        centre_y = (y + ch / 2) / roi.shape[0]
        if (
            1.20 <= aspect <= 12.0 and .0005 <= fraction <= .060
            and .30 <= centre_x <= .88 and .10 <= centre_y <= .80
        ):
            display_found = True
            break
    if not display_found:
        return False

    # Detect the actual 3x4 keypad grid using local bright-on-dark contrast.
    # Requiring both row and column regularity rejects green rifle skins and
    # green map textures, which otherwise resemble the C4 display colour.
    keypad_roi = image[int(h * .68):int(h * .995), int(w * .45):int(w * .74)]
    if keypad_roi.size == 0:
        return False
    gray = cv2.cvtColor(keypad_roi, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (290, 315), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    local_bright = cv2.morphologyEx(normalized, cv2.MORPH_TOPHAT, kernel)
    _threshold, bright = cv2.threshold(local_bright, 25, 255, cv2.THRESH_BINARY)
    contours, _hierarchy = cv2.findContours(
        bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    key_centres: list[tuple[float, float]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        width_ratio = cw / 290.0
        height_ratio = ch / 315.0
        centre_x = (x + cw / 2) / 290.0
        centre_y = (y + ch / 2) / 315.0
        fill = area / max(cw * ch, 1)
        if (
            .018 <= width_ratio <= .120
            and .015 <= height_ratio <= .100
            and .50 <= cw / max(ch, 1) <= 2.50
            and fill >= .25
            and .15 <= centre_x <= .90
            and .05 <= centre_y <= .95
        ):
            key_centres.append((centre_x, centre_y))
    x_groups = _coordinate_groups([point[0] for point in key_centres], .025)
    y_groups = _coordinate_groups([point[1] for point in key_centres], .035)
    return (
        len(key_centres) >= 10
        and sum(len(group) >= 3 for group in x_groups) >= 3
        and sum(len(group) >= 3 for group in y_groups) >= 4
    )


def _coordinate_groups(values: list[float], tolerance: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and abs(value - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups
