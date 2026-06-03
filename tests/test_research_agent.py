"""Test research agent for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ResearchAgentConfig:
    name: str = "research-agent"
    max_sources: int = 10
    depth: int = 2
    verbose: bool = False


@dataclass
class ResearchSource:
    id: str
    title: str
    content: str
    relevance_score: float = 0.0
    url: str = ""


@dataclass
class ResearchReport:
    topic: str
    summary: str
    sources: List[ResearchSource]
    findings: List[str]
    confidence: float = 0.0


class ResearchAgent:
    def __init__(self, config: ResearchAgentConfig = None):
        self._config = config or ResearchAgentConfig()
        self._search_results: List[ResearchSource] = []

    @property
    def config(self):
        return self._config

    def search(self, query: str, max_results: int = None) -> List[ResearchSource]:
        if not query:
            raise ValueError("Query cannot be empty")
        limit = max_results or self._config.max_sources
        results = [
            ResearchSource(id=f"src_{i}", title=f"Source {i}", content=f"Content about {query}", relevance_score=1.0 - i * 0.1)
            for i in range(min(limit, 5))
        ]
        self._search_results.extend(results)
        return results

    def analyze(self, sources: List[ResearchSource]) -> List[str]:
        if not sources:
            return []
        findings = []
        for src in sources:
            findings.append(f"Finding from {src.title}: {src.content[:50]}")
        return findings

    def synthesize(self, topic: str, findings: List[str], sources: List[ResearchSource]) -> ResearchReport:
        summary = f"Research summary on '{topic}': Found {len(findings)} findings from {len(sources)} sources."
        confidence = min(1.0, len(sources) / 5.0) if sources else 0.0
        return ResearchReport(
            topic=topic,
            summary=summary,
            sources=sources,
            findings=findings,
            confidence=confidence,
        )

    def research(self, topic: str) -> ResearchReport:
        if not topic:
            raise ValueError("Topic cannot be empty")
        sources = self.search(topic)
        findings = self.analyze(sources)
        return self.synthesize(topic, findings, sources)

    def get_search_history(self) -> List[ResearchSource]:
        return list(self._search_results)

    def clear_history(self):
        self._search_results.clear()


class TestResearchAgentConfig:
    def test_defaults(self):
        config = ResearchAgentConfig()
        assert config.max_sources == 10
        assert config.depth == 2

    def test_custom(self):
        config = ResearchAgentConfig(max_sources=5, depth=3)
        assert config.max_sources == 5


class TestResearchAgent:
    def test_search(self):
        agent = ResearchAgent()
        results = agent.search("machine learning")
        assert len(results) > 0
        assert all(isinstance(r, ResearchSource) for r in results)

    def test_search_empty_query(self):
        agent = ResearchAgent()
        with pytest.raises(ValueError, match="empty"):
            agent.search("")

    def test_search_max_results(self):
        agent = ResearchAgent()
        results = agent.search("test", max_results=2)
        assert len(results) <= 2

    def test_analyze(self):
        agent = ResearchAgent()
        sources = [ResearchSource(id="1", title="Test", content="ML content")]
        findings = agent.analyze(sources)
        assert len(findings) == 1

    def test_analyze_empty(self):
        agent = ResearchAgent()
        assert agent.analyze([]) == []

    def test_synthesize(self):
        agent = ResearchAgent()
        sources = [ResearchSource(id="1", title="Test", content="Content")]
        findings = ["Finding 1"]
        report = agent.synthesize("AI", findings, sources)
        assert report.topic == "AI"
        assert len(report.findings) == 1
        assert report.confidence > 0

    def test_research(self):
        agent = ResearchAgent()
        report = agent.research("neural networks")
        assert report.topic == "neural networks"
        assert len(report.sources) > 0
        assert report.summary

    def test_research_empty_topic(self):
        agent = ResearchAgent()
        with pytest.raises(ValueError, match="empty"):
            agent.research("")

    def test_search_history(self):
        agent = ResearchAgent()
        agent.search("topic1")
        agent.search("topic2")
        assert len(agent.get_search_history()) > 0

    def test_clear_history(self):
        agent = ResearchAgent()
        agent.search("test")
        agent.clear_history()
        assert len(agent.get_search_history()) == 0
