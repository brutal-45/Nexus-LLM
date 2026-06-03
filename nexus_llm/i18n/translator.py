"""Translator with pluralization and nested-key support."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from nexus_llm.i18n.manager import I18nManager


class Translator:
    """High-level translation helper bound to an I18nManager.

    Provides a concise ``t()`` shortcut, pluralization via
    ``pluralize()``, and automatic variable interpolation.

    Example::

        t = Translator(manager)
        t.t("greeting", locale="es", name="Mundo")   # "Hola, Mundo"
        t.pluralize("items", count=5, locale="en")   # "5 items"
    """

    # Suffixes used for pluralization lookup.
    _PLURAL_SUFFIXES = ("zero", "one", "two", "few", "many", "other")

    def __init__(self, manager: I18nManager) -> None:
        self._manager = manager

    # ------------------------------------------------------------------
    # Core translate shortcut
    # ------------------------------------------------------------------

    def t(self, key: str, locale: Optional[str] = None, **kwargs: Any) -> str:
        """Translate *key* in the given (or current) locale.

        This is a convenience wrapper around
        :py:meth:`I18nManager.translate` that also allows explicitly
        specifying the locale.

        Args:
            key: Dot-separated translation key.
            locale: Override locale; defaults to the manager's current.
            **kwargs: Variables for ``{name}``-style interpolation.

        Returns:
            The translated string.
        """
        saved: Optional[str] = None
        if locale is not None:
            saved = self._manager.get_locale()
            self._manager.set_locale(locale)

        try:
            return self._manager.translate(key, **kwargs)
        finally:
            if saved is not None:
                self._manager.set_locale(saved)

    # ------------------------------------------------------------------
    # Pluralization
    # ------------------------------------------------------------------

    def pluralize(self, key: str, count: int, locale: Optional[str] = None, **kwargs: Any) -> str:
        """Return a pluralized translation based on *count*.

        Looks up keys of the form ``key.zero``, ``key.one``,
        ``key.other`` (CLDR-style).  If a specific plural form is not
        found, falls back to ``key.other``, then to the base *key*.

        The ``{count}`` placeholder is automatically injected unless
        explicitly overridden via *kwargs*.

        Args:
            key: Base translation key.
            count: The numeric count driving plural form selection.
            locale: Override locale.
            **kwargs: Extra interpolation variables.

        Returns:
            The pluralized and interpolated string.
        """
        plural_form = self._select_plural_form(count, locale)
        kwargs.setdefault("count", count)

        # Try specific plural form → "other" → base key
        for candidate in (f"{key}.{plural_form}", f"{key}.other", key):
            result = self.t(candidate, locale=locale, **kwargs)
            if result != candidate:
                return result

        # Last resort: return "count key"
        return f"{count} {key}"

    # ------------------------------------------------------------------
    # Variable interpolation helpers (exposed for external use)
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate(template: str, **kwargs: Any) -> str:
        """Interpolate ``{name}``-style placeholders in *template*.

        Missing placeholders are left as-is rather than raising
        ``KeyError``.

        Args:
            template: String with ``{var}`` placeholders.
            **kwargs: Variable values.

        Returns:
            The interpolated string.
        """
        try:
            return template.format_map(_SafeDict(**kwargs))
        except (ValueError, IndexError):
            return template

    @staticmethod
    def extract_variables(template: str) -> list[str]:
        """Extract variable names from a template string.

        Args:
            template: String containing ``{var}`` placeholders.

        Returns:
            A deduplicated list of variable names, in order of
            first appearance.
        """
        seen: set[str] = set()
        result: list[str] = []
        for match in re.finditer(r"\{(\w+)\}", template):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_plural_form(count: int, locale: Optional[str] = None) -> str:
        """Select a CLDR-style plural form for *count*.

        This is a simplified implementation that covers the most common
        patterns for the built-in locales.  Full CLDR plural rules are
        locale-specific; here we use a reasonable heuristic.

        Args:
            count: The numeric count.
            locale: Optional locale hint.

        Returns:
            One of ``"zero"``, ``"one"``, ``"few"``, ``"many"``,
            ``"other"``.
        """
        if count == 0:
            return "zero"
        if count == 1:
            return "one"

        # Slavic languages (ru) and Arabic (ar) have "few" / "many"
        if locale in ("ru", "ar", "hi"):
            last_two = abs(count) % 100
            last_one = abs(count) % 10
            if 2 <= last_one <= 4 and not (12 <= last_two <= 14):
                return "few"
            if last_one == 0 or (5 <= last_one <= 9) or (11 <= last_two <= 14):
                return "many"

        return "other"


class _SafeDict(dict):  # noqa: D101
    """Dict that leaves unknown placeholders intact."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return f"{{{key}}}"
