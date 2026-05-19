"""
Nexus-LLM Internationalization (i18n) Module

Provides localization support for multiple languages.
Includes a Localizer class for loading and retrieving translated strings.
"""

from nexus_llm.i18n.localizer import (
    Localizer,
    LocalizationError,
    LocaleNotFoundError,
    get_localizer,
    set_locale,
    t,
)

__all__ = [
    "Localizer",
    "LocalizationError",
    "LocaleNotFoundError",
    "get_localizer",
    "set_locale",
    "t",
]
