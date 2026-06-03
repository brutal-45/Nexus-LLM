"""Nexus-LLM Search Tool.

Provides the SearchTool for searching through local documents and text
corpora using keyword matching and simple relevance scoring.
"""

import logging
import re
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """Search tool for finding text in local documents.

    Supports keyword search, regex search, and fuzzy matching
    over a collection of text documents stored in memory.

    Example::

        search = SearchTool()
        search.index("doc1", "The quick brown fox jumps over the lazy dog")
        result = search.execute(query="fox", method="keyword")
    """

    def __init__(self) -> None:
        super().__init__(name="search", description="Search through indexed documents")
        self._documents: Dict[str, str] = {}

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type=ParameterType.STRING,
                description="Search query string",
                required=True,
            ),
            ToolParameter(
                name="method",
                type=ParameterType.STRING,
                description="Search method",
                required=False,
                default="keyword",
                choices=["keyword", "regex", "fuzzy"],
            ),
            ToolParameter(
                name="max_results",
                type=ParameterType.INTEGER,
                description="Maximum number of results",
                required=False,
                default=10,
            ),
        ]

    def index(self, doc_id: str, content: str) -> None:
        """Add a document to the search index.

        Args:
            doc_id: Unique document identifier.
            content: Document text content.
        """
        self._documents[doc_id] = content

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the index.

        Args:
            doc_id: Document identifier.

        Returns:
            True if the document was found and removed.
        """
        return self._documents.pop(doc_id, None) is not None

    def execute(self, query: str = "", method: str = "keyword", max_results: int = 10, **kwargs: Any) -> ToolResult:
        """Execute a search query.

        Args:
            query: Search query string.
            method: Search method (keyword, regex, fuzzy).
            max_results: Maximum results to return.

        Returns:
            ToolResult with matching documents and scores.
        """
        if not query:
            return ToolResult(tool_name=self.name, success=False, error="Empty query")
        if not self._documents:
            return ToolResult(tool_name=self.name, success=False, error="No documents indexed")

        if method == "keyword":
            results = self._keyword_search(query, max_results)
        elif method == "regex":
            results = self._regex_search(query, max_results)
        elif method == "fuzzy":
            results = self._fuzzy_search(query, max_results)
        else:
            return ToolResult(tool_name=self.name, success=False, error=f"Unknown method: {method}")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=results,
            metadata={"query": query, "method": method, "total_indexed": len(self._documents)},
        )

    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using keyword matching with TF-based scoring."""
        query_terms = set(query.lower().split())
        results: List[Dict[str, Any]] = []

        for doc_id, content in self._documents.items():
            content_lower = content.lower()
            content_terms = content_lower.split()
            matches = sum(1 for t in content_terms if t in query_terms)
            if matches > 0:
                score = matches / len(content_terms) if content_terms else 0
                results.append({"doc_id": doc_id, "score": round(score, 4), "matches": matches})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max_results]

    def _regex_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using regular expression matching."""
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            return [{"error": f"Invalid regex: {exc}"}]

        results: List[Dict[str, Any]] = []
        for doc_id, content in self._documents.items():
            matches = pattern.findall(content)
            if matches:
                results.append({"doc_id": doc_id, "match_count": len(matches), "matches": matches[:5]})

        results.sort(key=lambda r: r["match_count"], reverse=True)
        return results[:max_results]

    def _fuzzy_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using simple fuzzy matching (Levenshtein-inspired)."""
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()

        for doc_id, content in self._documents.items():
            score = self._similarity(query_lower, content.lower())
            if score > 0.1:
                results.append({"doc_id": doc_id, "score": round(score, 4)})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max_results]

    @staticmethod
    def _similarity(query: str, text: str) -> float:
        """Compute a simple similarity score between query and text."""
        query_words = set(query.split())
        text_words = set(text.split())
        if not query_words:
            return 0.0
        intersection = query_words & text_words
        return len(intersection) / len(query_words)
