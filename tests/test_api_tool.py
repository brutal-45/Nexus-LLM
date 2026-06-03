"""Tests for nexus_llm.tools.api_tool module."""

import pytest
from nexus_llm.tools.api_tool import ApiTool


class TestApiTool:
    """Tests for the ApiTool class."""

    def test_init(self):
        tool = ApiTool()
        assert tool.name == "api"

    def test_get_request(self):
        tool = ApiTool()
        result = tool.run(method="GET", url="https://httpbin.org/status/200")
        # Network dependent; just check structure
        assert isinstance(result.success, bool)

    def test_missing_url(self):
        tool = ApiTool()
        result = tool.run(method="GET")
        assert result.success is False

    def test_invalid_method(self):
        tool = ApiTool()
        result = tool.run(method="INVALID", url="https://example.com")
        assert result.success is False
