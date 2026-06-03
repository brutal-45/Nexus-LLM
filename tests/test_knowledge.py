"""Tests for the knowledge module.

Covers KnowledgeBase, KnowledgeGraph, and KnowledgeEntry.
"""

from __future__ import annotations

import pytest

from nexus_llm.knowledge.base import KnowledgeBase
from nexus_llm.knowledge.graph import KnowledgeGraph
from nexus_llm.knowledge.entry import KnowledgeEntry


# ---------------------------------------------------------------------------
# KnowledgeEntry
# ---------------------------------------------------------------------------

class TestKnowledgeEntry:
    """Tests for KnowledgeEntry."""

    def test_create_entry(self):
        entry = KnowledgeEntry(
            id="e1",
            title="Python",
            content="Python is a programming language",
        )
        assert entry.id == "e1"
        assert entry.title == "Python"

    def test_entry_defaults(self):
        entry = KnowledgeEntry(id="e2", title="Test")
        assert entry.content == ""
        assert entry.tags == []
        assert entry.metadata == {}

    def test_to_dict(self):
        entry = KnowledgeEntry(id="e1", title="Test", content="Content")
        d = entry.to_dict()
        assert d["id"] == "e1"
        assert d["title"] == "Test"

    def test_from_dict(self):
        data = {"id": "e1", "title": "Test", "content": "Hello", "tags": ["a"], "metadata": {}}
        entry = KnowledgeEntry.from_dict(data)
        assert entry.id == "e1"
        assert entry.tags == ["a"]


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    """Tests for KnowledgeBase."""

    def test_add_and_get(self):
        kb = KnowledgeBase()
        entry = KnowledgeEntry(id="e1", title="Python", content="Programming language")
        kb.add(entry)
        retrieved = kb.get("e1")
        assert retrieved is not None
        assert retrieved.title == "Python"

    def test_get_nonexistent(self):
        kb = KnowledgeBase()
        assert kb.get("nonexistent") is None

    def test_search(self):
        kb = KnowledgeBase()
        kb.add(KnowledgeEntry(id="e1", title="Python", content="A programming language"))
        kb.add(KnowledgeEntry(id="e2", title="Java", content="Another programming language"))
        kb.add(KnowledgeEntry(id="e3", title="Cooking", content="Italian recipes"))
        results = kb.search("programming")
        assert len(results) >= 2

    def test_delete(self):
        kb = KnowledgeBase()
        kb.add(KnowledgeEntry(id="e1", title="Test"))
        assert kb.delete("e1") is True
        assert kb.delete("e1") is False

    def test_list_entries(self):
        kb = KnowledgeBase()
        kb.add(KnowledgeEntry(id="e1", title="A"))
        kb.add(KnowledgeEntry(id="e2", title="B"))
        entries = kb.list_entries()
        assert len(entries) == 2

    def test_update(self):
        kb = KnowledgeBase()
        kb.add(KnowledgeEntry(id="e1", title="Old", content="Old content"))
        kb.add(KnowledgeEntry(id="e1", title="New", content="New content"))
        retrieved = kb.get("e1")
        assert retrieved.title == "New"

    def test_count(self):
        kb = KnowledgeBase()
        kb.add(KnowledgeEntry(id="e1", title="A"))
        kb.add(KnowledgeEntry(id="e2", title="B"))
        assert kb.count() == 2


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def test_add_node(self):
        kg = KnowledgeGraph()
        kg.add_node("python", {"type": "language"})
        assert kg.has_node("python")

    def test_add_edge(self):
        kg = KnowledgeGraph()
        kg.add_node("python")
        kg.add_node("django")
        kg.add_edge("python", "django", "framework_for")
        assert kg.has_edge("python", "django")

    def test_get_neighbors(self):
        kg = KnowledgeGraph()
        kg.add_node("python")
        kg.add_node("django")
        kg.add_node("flask")
        kg.add_edge("python", "django", "framework_for")
        kg.add_edge("python", "flask", "framework_for")
        neighbors = kg.get_neighbors("python")
        assert len(neighbors) >= 2

    def test_remove_node(self):
        kg = KnowledgeGraph()
        kg.add_node("python")
        kg.remove_node("python")
        assert not kg.has_node("python")

    def test_get_node_data(self):
        kg = KnowledgeGraph()
        kg.add_node("python", {"type": "language"})
        data = kg.get_node_data("python")
        assert data["type"] == "language"

    def test_search(self):
        kg = KnowledgeGraph()
        kg.add_node("python", {"type": "language"})
        kg.add_node("java", {"type": "language"})
        kg.add_node("cooking", {"type": "hobby"})
        results = kg.search("language")
        assert len(results) >= 2
