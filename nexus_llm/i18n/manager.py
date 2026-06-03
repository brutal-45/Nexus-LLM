"""Central i18n manager for locale and translation management."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from nexus_llm.i18n.translator import Translator
from nexus_llm.i18n.locale import Locale, BUILTIN_LOCALES


class I18nManager:
    """Manages application-wide internationalization settings.

    Provides a unified interface for setting the active locale,
    registering translation dictionaries, and resolving translated
    strings with variable interpolation.

    Example::

        manager = I18nManager()
        manager.register_translations("es", {
            "greeting": "Hola, {name}",
            "model.loading.progress": "Cargando modelo... {percent}%",
        })
        manager.set_locale("es")
        manager.translate("greeting", name="Mundo")  # "Hola, Mundo"
    """

    _global_instance: Optional[I18nManager] = None
    _instance_lock = threading.Lock()

    def __init__(self, default_locale: str = "en") -> None:
        self._default_locale = default_locale
        self._current_locale = default_locale
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._locales: Dict[str, Locale] = {
            loc.code: loc for loc in BUILTIN_LOCALES
        }
        self._lock = threading.RLock()
        self._translator = Translator(self)

        # Seed with empty translation dict for each built-in locale
        for code in self._locales:
            self._translations.setdefault(code, {})

    # ------------------------------------------------------------------
    # Class-level singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> I18nManager:
        """Return the global I18nManager singleton (lazy-created)."""
        if cls._global_instance is None:
            with cls._instance_lock:
                if cls._global_instance is None:
                    cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the global singleton (useful for testing)."""
        with cls._instance_lock:
            cls._global_instance = None

    # ------------------------------------------------------------------
    # Locale management
    # ------------------------------------------------------------------

    def set_locale(self, locale_code: str) -> None:
        """Set the active locale.

        Args:
            locale_code: An ISO 639-1 language code (e.g. ``"en"``, ``"es"``).

        Raises:
            ValueError: If the locale code is not registered.
        """
        with self._lock:
            if locale_code not in self._locales:
                raise ValueError(
                    f"Unknown locale '{locale_code}'. "
                    f"Available: {', '.join(sorted(self._locales))}"
                )
            self._current_locale = locale_code

    def get_locale(self) -> str:
        """Return the currently active locale code."""
        return self._current_locale

    def list_locales(self) -> list[str]:
        """Return a sorted list of all registered locale codes."""
        with self._lock:
            return sorted(self._locales.keys())

    # ------------------------------------------------------------------
    # Translation registration
    # ------------------------------------------------------------------

    def register_translations(
        self, locale_code: str, translations: Dict[str, Any]
    ) -> None:
        """Register a flat or nested translation dictionary for a locale.

        Nested dictionaries are flattened using dot-separated keys so that
        ``{"model": {"loading": {"progress": "Loading…"}}}`` becomes
        accessible as ``"model.loading.progress"``.

        Args:
            locale_code: The locale the translations belong to.
            translations: A dict mapping keys to translated strings (or
                nested dicts of the same).
        """
        with self._lock:
            # Auto-register locale if it's a known built-in
            if locale_code not in self._locales and locale_code in {
                loc.code for loc in BUILTIN_LOCALES
            }:
                for loc in BUILTIN_LOCALES:
                    if loc.code == locale_code:
                        self._locales[locale_code] = loc
                        break

            flat = self._flatten(translations)
            self._translations.setdefault(locale_code, {}).update(flat)

    # ------------------------------------------------------------------
    # Translation lookup
    # ------------------------------------------------------------------

    def translate(self, key: str, **kwargs: Any) -> str:
        """Translate *key* in the current locale, interpolating *kwargs*.

        Falls back to the default locale if the key is missing in the
        current locale, and returns the raw key if not found anywhere.

        Args:
            key: Dot-separated translation key.
            **kwargs: Variables for ``{name}``-style interpolation.

        Returns:
            The translated (and interpolated) string.
        """
        with self._lock:
            locale = self._current_locale
            template = self._translations.get(locale, {}).get(key)

            # Fallback chain: current locale → default locale → raw key
            if template is None and locale != self._default_locale:
                template = self._translations.get(
                    self._default_locale, {}
                ).get(key)

            if template is None:
                return key

            if isinstance(template, str):
                try:
                    return template.format_map(
                        _SafeFormatDict(**kwargs)
                    )
                except (KeyError, IndexError):
                    return template

            return str(template)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def translator(self) -> Translator:
        """Return the associated Translator instance."""
        return self._translator

    def get_translations(self, locale_code: Optional[str] = None) -> Dict[str, str]:
        """Return the flat translation map for a locale."""
        with self._lock:
            code = locale_code or self._current_locale
            return dict(self._translations.get(code, {}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(
        d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Recursively flatten a nested dict into dot-separated keys."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    I18nManager._flatten(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)


class _SafeFormatDict(dict):  # noqa: D101
    """Dict subclass that returns ``{key}`` unchanged for missing keys.

    This prevents ``KeyError`` during ``str.format_map()`` when a
    variable is not supplied, keeping the placeholder visible.
    """

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
