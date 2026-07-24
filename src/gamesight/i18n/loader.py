"""Lightweight i18n framework for GameSight.

Design
------
- Translations stored as flat JSON files under ``locales/``.
- ``I18nLoader`` loads one locale at a time and exposes a ``t(key, **kwargs)``
  method for key-path lookups with optional ``str.format`` interpolation.
- Locale is selected at init time; switching locale creates a new loader.
- Missing keys fall back to the key path itself so the UI never breaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_LOCALE_DIR = Path(__file__).resolve().parent / "locales"
_AVAILABLE_LOCALES = {"en", "zh-CN"}
_DEFAULT_LOCALE = "en"


class I18nLoader:
    """Load and serve translations for one locale.

    Parameters
    ----------
    locale:
        Locale code (``"en"`` or ``"zh-CN"``).
    locale_dir:
        Override the default ``locales/`` directory (useful for testing).
    """

    def __init__(
        self,
        locale: str = _DEFAULT_LOCALE,
        locale_dir: Path | None = None,
    ) -> None:
        if locale not in _AVAILABLE_LOCALES:
            raise ValueError(
                f"Unsupported locale '{locale}'. Available: {sorted(_AVAILABLE_LOCALES)}"
            )
        self._locale = locale
        self._dir = locale_dir or _LOCALE_DIR
        self._data: dict[str, Any] = {}
        self._load()

    # -- public API -----------------------------------------------------------

    def t(self, key: str, **kwargs: str | int | float) -> str:
        """Look up a translation by dot-separated key path.

        ``key`` uses dotted notation: ``"overview.match_overview"``.

        ``**kwargs`` are interpolated into the string via ``str.format``,
        e.g. ``t("run.processing", w=1920, h=1080, fps=60)``.
        """
        value = self._data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return key  # graceful fallback
            value = value[part]

        if not isinstance(value, str):
            return key

        if kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                return value

        return value

    @property
    def locale(self) -> str:
        return self._locale

    @property
    def lang_name(self) -> str:
        return str(self._data.get("lang_name", self._locale))

    @staticmethod
    def available_locales() -> set[str]:
        return _AVAILABLE_LOCALES.copy()

    # -- internal -------------------------------------------------------------

    def _load(self) -> None:
        path = self._dir / f"{self._locale}.json"
        if not path.exists():
            raise FileNotFoundError(f"Locale file not found: {path}")
        self._data = json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_current: I18nLoader | None = None


def get_loader(locale: str | None = None) -> I18nLoader:
    """Return the current or a new I18nLoader for *locale*."""
    global _current
    if locale is not None:
        _current = I18nLoader(locale)
    if _current is None:
        _current = I18nLoader(_DEFAULT_LOCALE)
    return _current


def t(key: str, **kwargs: str | int | float) -> str:
    """Convenience: call ``t()`` on the current loader."""
    return get_loader().t(key, **kwargs)
