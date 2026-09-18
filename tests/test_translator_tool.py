"""Tests for nexus_llm.tools.translator module."""

import pytest
from nexus_llm.tools.translator import TranslatorTool


class TestTranslatorTool:
    def test_init(self):
        tool = TranslatorTool()
        assert tool.name == "translator"

    def test_translate(self):
        tool = TranslatorTool()
        result = tool.run(text="Hello world", source_lang="en", target_lang="es")
        assert result.success is True
        assert "translated_text" in result.output

    def test_translate_empty_text(self):
        tool = TranslatorTool()
        result = tool.run(text="", source_lang="en", target_lang="es")
        assert result.success is False

    def test_translate_no_target(self):
        tool = TranslatorTool()
        result = tool.run(text="Hello", source_lang="en")
        assert result.success is False

    def test_get_supported_languages(self):
        tool = TranslatorTool()
        langs = tool.get_supported_languages()
        assert "en" in langs
        assert "es" in langs
        assert len(langs) > 5
