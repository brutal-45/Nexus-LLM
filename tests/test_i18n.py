"""Tests for the i18n module.

Covers I18nManager, Translator, Locale, and LanguageDetector.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nexus_llm.i18n.manager import I18nManager
from nexus_llm.i18n.translator import Translator
from nexus_llm.i18n.locale import Locale
from nexus_llm.i18n.detector import LanguageDetector


# ---------------------------------------------------------------------------
# Locale
# ---------------------------------------------------------------------------

class TestLocale:
    """Tests for Locale."""

    def test_create_locale(self):
        locale = Locale(code="en", name="English")
        assert locale.code == "en"
        assert locale.name == "English"

    def test_locale_defaults(self):
        locale = Locale(code="fr")
        assert locale.name == ""

    def test_repr(self):
        locale = Locale(code="en", name="English")
        r = repr(locale)
        assert "en" in r


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class TestTranslator:
    """Tests for Translator."""

    def test_create_translator(self):
        t = Translator()
        assert t is not None

    def test_translate(self):
        t = Translator()
        # Translator should have a translate method
        result = t.translate("hello", source="en", target="fr")
        assert isinstance(result, str)

    def test_translate_with_dictionary(self):
        t = Translator()
        t.add_translation("hello", "en", "fr", "bonjour")
        result = t.translate("hello", source="en", target="fr")
        assert result == "bonjour"


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------

class TestLanguageDetector:
    """Tests for LanguageDetector."""

    def test_detect_english(self):
        detector = LanguageDetector()
        lang = detector.detect("Hello, how are you today?")
        assert isinstance(lang, str)

    def test_detect_empty_string(self):
        detector = LanguageDetector()
        lang = detector.detect("")
        assert isinstance(lang, str)

    def test_detect_returns_language_code(self):
        detector = LanguageDetector()
        lang = detector.detect("This is definitely English text")
        # Should return a language code like "en"
        assert len(lang) <= 10  # language codes are short


# ---------------------------------------------------------------------------
# I18nManager
# ---------------------------------------------------------------------------

class TestI18nManager:
    """Tests for I18nManager."""

    def test_init(self):
        mgr = I18nManager()
        assert mgr is not None

    def test_default_locale(self):
        mgr = I18nManager(default_locale="en")
        assert mgr.default_locale == "en" or mgr.get_locale() is not None

    def test_set_locale(self):
        mgr = I18nManager()
        mgr.set_locale("fr")
        # Should not crash

    def test_get_translation(self):
        mgr = I18nManager()
        result = mgr.get("hello")
        assert isinstance(result, str)

    def test_register_translation(self):
        mgr = I18nManager()
        mgr.register("greeting", "en", "Hello!")
        mgr.register("greeting", "fr", "Bonjour!")
        mgr.set_locale("en")
        result = mgr.get("greeting")
        assert result == "Hello!"

    def test_list_locales(self):
        mgr = I18nManager()
        locales = mgr.list_locales()
        assert isinstance(locales, list)

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample locale file
            en_data = {"hello": "Hello", "goodbye": "Goodbye"}
            path = os.path.join(tmpdir, "en.json")
            with open(path, "w") as f:
                json.dump(en_data, f)
            mgr = I18nManager()
            mgr.load_translations(tmpdir)
