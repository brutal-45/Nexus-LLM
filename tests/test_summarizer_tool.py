"""Tests for nexus_llm.tools.summarizer module."""

import pytest
from nexus_llm.tools.summarizer import SummarizerTool


class TestSummarizerTool:
    """Tests for the SummarizerTool class."""

    def test_init(self):
        tool = SummarizerTool()
        assert tool.name == "summarizer"

    def test_summarize(self):
        tool = SummarizerTool()
        result = tool.run(
            text="This is a long piece of text. It contains multiple sentences. " * 10,
            max_length=50,
        )
        assert result.success is True
        assert isinstance(result.output, str)

    def test_summarize_empty(self):
        tool = SummarizerTool()
        result = tool.run(text="", max_length=50)
        assert result.success is False

    def test_summarize_short_text(self):
        tool = SummarizerTool()
        result = tool.run(text="Short text.", max_length=100)
        assert result.success is True
