"""Nexus-LLM Web Scraper Tool.

Provides the WebScraperTool for fetching and extracting content from
web pages, with configurable timeout, headers, and extraction modes.
"""

import logging
import re
from typing import Any, Dict, List
from html.parser import HTMLParser

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class _TextExtractor(HTMLParser):
    """Simple HTML to text extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False
        if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._text_parts.append(data)

    def get_text(self) -> str:
        return re.sub(r'\n{3,}', '\n\n', ''.join(self._text_parts)).strip()


class WebScraperTool(BaseTool):
    """Web scraper tool for fetching and extracting page content.

    Uses urllib from the standard library to fetch pages and a
    built-in HTML parser to extract text content.

    Example::

        scraper = WebScraperTool()
        result = scraper.execute(url="https://example.com")
    """

    def __init__(self, default_timeout: int = 30, user_agent: str = "") -> None:
        super().__init__(name="web_scraper", description="Fetch and extract content from web pages")
        self._timeout = default_timeout
        self._user_agent = user_agent or "Nexus-LLM/1.0 WebScraper"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", type=ParameterType.STRING, description="URL to fetch", required=True),
            ToolParameter(name="mode", type=ParameterType.STRING, description="Extraction mode", required=False, default="text", choices=["text", "raw", "links"]),
            ToolParameter(name="timeout", type=ParameterType.INTEGER, description="Request timeout in seconds", required=False, default=30),
        ]

    def execute(self, url: str = "", mode: str = "text", timeout: int = 30, **kwargs: Any) -> ToolResult:
        """Fetch and extract content from a URL.

        Args:
            url: The URL to fetch.
            mode: Extraction mode (text, raw, links).
            timeout: Request timeout in seconds.

        Returns:
            ToolResult with extracted content.
        """
        if not url:
            return ToolResult(tool_name=self.name, success=False, error="No URL provided")

        try:
            import urllib.request
            import urllib.error

            request = urllib.request.Request(
                url,
                headers={"User-Agent": self._user_agent},
            )
            with urllib.request.urlopen(request, timeout=timeout or self._timeout) as response:
                html = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return ToolResult(tool_name=self.name, success=False, error=f"HTTP {exc.code}: {exc.reason}")
        except urllib.error.URLError as exc:
            return ToolResult(tool_name=self.name, success=False, error=f"URL error: {exc.reason}")
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=f"Fetch error: {exc}")

        if mode == "raw":
            return ToolResult(tool_name=self.name, success=True, output=html, metadata={"url": url, "mode": mode})

        if mode == "links":
            links = re.findall(r'href=["\'](https?://[^"\']+)', html)
            return ToolResult(tool_name=self.name, success=True, output=links, metadata={"url": url, "link_count": len(links)})

        # Default: text extraction
        extractor = _TextExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        text = extractor.get_text()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=text,
            metadata={"url": url, "mode": mode, "text_length": len(text)},
        )
