"""Internationalization (i18n) support for Nexus-LLM.

Provides locale management, translation, pluralization,
and language detection capabilities.
"""

from nexus_llm.i18n.detector import LanguageDetector
from nexus_llm.i18n.locale import Locale
from nexus_llm.i18n.manager import I18nManager
from nexus_llm.i18n.translator import Translator

__all__ = [
    "I18nManager",
    "LanguageDetector",
    "Locale",
    "Translator",
]
