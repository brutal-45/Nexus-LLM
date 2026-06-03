"""Tests for nexus_llm.tools.search module."""

import pytest
from nexus_llm.tools.search import SearchTool


class TestSearchTool:
    """Tests for the SearchTool class."""

    def test_init(self):
        tool = SearchTool()
        assert tool.name == "search"

    def test_search_basic(self):
        tool = SearchTool()
        result = tool.run(query="python", max_results=5)
        assert result.success is True

    def test_search_with_filters(self):
        tool = SearchTool()
        result = tool.run(query="machine learning", filters={"category": "tech"})
        assert result.success is True

    def test_search_empty_query(self):
        tool = SearchTool()
        result = tool.run(query="")
        assert result.success is False

    def test_search_max_results(self):
        tool = SearchTool()
        result = tool.run(query="test", max_results=3)
        assert result.success is True
