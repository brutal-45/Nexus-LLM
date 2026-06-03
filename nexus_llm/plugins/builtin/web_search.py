"""Web search plugin providing simulated web search functionality.

A builtin plugin that simulates web search results for testing
and development. In production, would integrate with real APIs.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class WebSearchPlugin:
    """Plugin providing web search functionality (simulated).

    Simulates web search with a curated knowledge base and
    generated results for unknown queries.
    """

    name = "web_search"
    version = "1.0.0"
    description = "Search the web for information (simulated data)"
    dependencies: List[str] = []
    tags = ["search", "web", "information", "builtin"]

    # Simulated knowledge base
    KNOWLEDGE_BASE = {
        "python": [
            {
                "title": "Python (programming language) - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "snippet": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            },
            {
                "title": "Welcome to Python.org",
                "url": "https://www.python.org",
                "snippet": "The official home of the Python Programming Language. Download Python, documentation, and community resources.",
            },
        ],
        "machine learning": [
            {
                "title": "Machine Learning - Stanford Online",
                "url": "https://www.coursera.org/learn/machine-learning",
                "snippet": "Machine learning is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence.",
            },
            {
                "title": "What is Machine Learning? - IBM",
                "url": "https://www.ibm.com/topics/machine-learning",
                "snippet": "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate human learning.",
            },
        ],
        "rag": [
            {
                "title": "Retrieval-Augmented Generation - Meta AI",
                "url": "https://ai.meta.com/blog/retrieval-augmented-generation/",
                "snippet": "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.",
            },
        ],
        "artificial intelligence": [
            {
                "title": "Artificial Intelligence - MIT Technology Review",
                "url": "https://www.technologyreview.com/topic/artificial-intelligence/",
                "snippet": "Artificial intelligence is the simulation of human intelligence processes by computer systems, including learning, reasoning, and self-correction.",
            },
        ],
        "transformer": [
            {
                "title": "Attention Is All You Need - arXiv",
                "url": "https://arxiv.org/abs/1706.03762",
                "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer.",
            },
        ],
    }

    def __init__(self, hook_manager: Optional[HookManager] = None, **kwargs):
        self.hook_manager = hook_manager
        self._active = False
        self._search_history: List[Dict[str, Any]] = []

    def activate(self) -> None:
        """Activate the web search plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="web_search_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("Web search plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the web search plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Web search plugin deactivated.")

    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a web search (simulated).

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            Dictionary with search results.
        """
        query = query.strip()
        if not query:
            return {"success": False, "error": "Empty query", "results": []}

        start_time = time.time()
        results = self._find_results(query, num_results)
        elapsed = time.time() - start_time

        search_record = {
            "query": query,
            "num_results": len(results),
            "elapsed_time": round(elapsed, 3),
            "timestamp": time.time(),
        }
        self._search_history.append(search_record)

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results),
            "elapsed_time": round(elapsed, 3),
        }

    def _find_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Find results for a search query."""
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []
        seen_urls = set()

        # Search knowledge base
        for topic, entries in self.KNOWLEDGE_BASE.items():
            topic_words = topic.split()
            query_words = query_lower.split()

            # Check for topic overlap
            overlap = sum(1 for w in topic_words if w in query_lower)
            overlap += sum(1 for w in query_words if w in topic)

            if overlap > 0:
                for entry in entries:
                    if entry["url"] not in seen_urls:
                        seen_urls.add(entry["url"])
                        result = entry.copy()
                        result["relevance_score"] = min(1.0, overlap * 0.3)
                        result["source"] = "knowledge_base"
                        results.append(result)

        # Check snippet content for additional matches
        for topic, entries in self.KNOWLEDGE_BASE.items():
            for entry in entries:
                if entry["url"] in seen_urls:
                    continue
                if any(word in entry["snippet"].lower() for word in query_words if len(word) > 2):
                    seen_urls.add(entry["url"])
                    result = entry.copy()
                    result["relevance_score"] = 0.3
                    result["source"] = "knowledge_base"
                    results.append(result)

        # Generate simulated results if not enough from knowledge base
        if len(results) < num_results:
            simulated = self._generate_simulated_results(query, num_results - len(results))
            for r in simulated:
                if r["url"] not in seen_urls:
                    results.append(r)

        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:num_results]

    def _generate_simulated_results(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Generate simulated search results for unknown queries."""
        hash_val = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        domains = ["example.com", "info-source.org", "knowledge-base.net", "reference.io", "data-hub.com"]

        results = []
        for i in range(count):
            domain = domains[(hash_val + i) % len(domains)]
            slug = query.lower().replace(" ", "-")
            url = f"https://{domain}/articles/{slug}-{i + 1}"

            results.append({
                "title": f"Information about {query.title()} - {domain}",
                "url": url,
                "snippet": (
                    f"This article provides detailed information about {query}. "
                    f"It covers key concepts, applications, and recent developments "
                    f"in the field of {query.lower()}."
                ),
                "relevance_score": 0.2 - (i * 0.05),
                "source": "simulated",
            })

        return results

    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent search history."""
        return self._search_history[-limit:]

    def clear_history(self) -> None:
        """Clear search history."""
        self._search_history.clear()

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for web search."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "search":
            query = kwargs.get("query", kwargs.get("expression", ""))
            if query:
                search_result = self.search(query)
                if search_result["success"]:
                    output_parts = []
                    for r in search_result["results"]:
                        output_parts.append(f"- {r['title']}\n  {r['snippet']}\n  URL: {r['url']}")
                    return "\n\n".join(output_parts)
                return f"Search error: {search_result.get('error', 'Unknown error')}"
        return result
