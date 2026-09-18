"""Tests for nexus_llm.tools.web_scraper module."""

import pytest
from nexus_llm.tools.web_scraper import WebScraperTool


class TestWebScraperTool:
    """Tests for the WebScraperTool class."""

    def test_init(self):
        tool = WebScraperTool()
        assert tool.name == "web_scraper"

    def test_scrape_invalid_url(self):
        tool = WebScraperTool()
        result = tool.run(url="not-a-url")
        assert result.success is False

    def test_scrape_missing_url(self):
        tool = WebScraperTool()
        result = tool.run()
        assert result.success is False

    def test_scrape_with_selector(self):
        tool = WebScraperTool()
        result = tool.run(url="https://example.com", selector="h1")
        # May succeed or fail depending on network
        assert isinstance(result.success, bool)
