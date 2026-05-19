"""Tests for nexus_llm.nexus.analyzer module."""

import pytest
from nexus_llm.nexus.analyzer import NexusAnalyzer


class TestNexusAnalyzer:
    """Tests for the NexusAnalyzer class."""

    def test_init_default(self):
        analyzer = NexusAnalyzer()
        assert analyzer is not None

    def test_analyze_text(self):
        analyzer = NexusAnalyzer()
        result = analyzer.analyze("Hello world, this is a test.")
        assert isinstance(result, dict)
        assert "word_count" in result

    def test_analyze_empty_text(self):
        analyzer = NexusAnalyzer()
        result = analyzer.analyze("")
        assert result["word_count"] == 0

    def test_analyze_sentiment(self):
        analyzer = NexusAnalyzer()
        result = analyzer.analyze_sentiment("I love this amazing product!")
        assert isinstance(result, dict)

    def test_analyze_readability(self):
        analyzer = NexusAnalyzer()
        result = analyzer.analyze_readability(
            "The quick brown fox jumps over the lazy dog. "
            "This is a simple sentence for testing readability."
        )
        assert isinstance(result, dict)

    def test_extract_keywords(self):
        analyzer = NexusAnalyzer()
        keywords = analyzer.extract_keywords(
            "Machine learning and artificial intelligence are transforming technology.",
            top_n=3,
        )
        assert isinstance(keywords, list)
        assert len(keywords) <= 3

    def test_get_statistics(self):
        analyzer = NexusAnalyzer()
        stats = analyzer.get_statistics("Hello world, this is a test sentence.")
        assert "char_count" in stats
        assert "word_count" in stats
        assert "sentence_count" in stats
