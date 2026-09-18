"""Tests for the memory module.

Covers ConversationMemory, SummaryMemory, WorkingMemory, and MemoryManager.
"""

from __future__ import annotations

import pytest

from nexus_llm.memory.conversation import ConversationMemory, Message
from nexus_llm.memory.summary import SummaryMemory, SummaryNode
from nexus_llm.memory.working import WorkingMemory
from nexus_llm.memory.manager import MemoryManager, MemoryNotFoundError


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_add_and_get_messages(self):
        mem = ConversationMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi there!")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_get_recent(self):
        mem = ConversationMemory()
        for i in range(10):
            mem.add_message("user", f"Message {i}")
        recent = mem.get_messages(n=3)
        assert len(recent) == 3

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_message("user", "Hello")
        mem.clear()
        assert len(mem.get_messages()) == 0

    def test_max_messages(self):
        mem = ConversationMemory(max_messages=5)
        for i in range(10):
            mem.add_message("user", f"Message {i}")
        assert len(mem.get_messages()) == 5

    def test_get_context(self):
        mem = ConversationMemory()
        mem.add_message("user", "Tell me about Python")
        mem.add_message("assistant", "Python is a programming language")
        mem.add_message("user", "What about Java?")
        context = mem.get_context("Python")
        assert len(context) >= 1

    def test_len(self):
        mem = ConversationMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi")
        assert len(mem) == 2

    def test_repr(self):
        mem = ConversationMemory()
        r = repr(mem)
        assert "ConversationMemory" in r

    def test_get_all(self):
        mem = ConversationMemory()
        mem.add_message("user", "A")
        mem.add_message("assistant", "B")
        all_msgs = mem.get_all()
        assert len(all_msgs) == 2

    def test_invalid_max_messages(self):
        with pytest.raises(ValueError):
            ConversationMemory(max_messages=0)

    def test_message_to_dict(self):
        mem = ConversationMemory()
        msg = mem.add_message("user", "Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"


# ---------------------------------------------------------------------------
# SummaryMemory
# ---------------------------------------------------------------------------

class TestSummaryMemory:
    """Tests for SummaryMemory."""

    def test_add(self):
        mem = SummaryMemory()
        node = mem.add("This is a conversation about AI.")
        assert isinstance(node, SummaryNode)
        assert node.content == "This is a conversation about AI."

    def test_get_summary(self):
        mem = SummaryMemory()
        mem.add("Message one")
        summary = mem.get_summary()
        assert isinstance(summary, str)
        assert "Message one" in summary

    def test_clear(self):
        mem = SummaryMemory()
        mem.add("Hello")
        mem.clear()
        assert len(mem) == 0

    def test_auto_summarization(self):
        mem = SummaryMemory(summarize_threshold=3)
        for i in range(3):
            mem.add(f"Message {i}")
        # After 3 entries, auto-summarization should kick in
        entries = mem.get_all_entries()
        assert len(entries) >= 1

    def test_get_all_entries(self):
        mem = SummaryMemory()
        mem.add("Entry 1")
        mem.add("Entry 2")
        entries = mem.get_all_entries()
        assert len(entries) == 2

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            SummaryMemory(summarize_threshold=1)

    def test_custom_summarizer(self):
        def my_summarizer(texts):
            return "CUSTOM: " + " | ".join(texts)

        mem = SummaryMemory(summarize_threshold=2, summarizer=my_summarizer)
        mem.add("A")
        mem.add("B")
        summary = mem.get_summary()
        assert "CUSTOM:" in summary

    def test_len(self):
        mem = SummaryMemory()
        mem.add("A")
        mem.add("B")
        assert len(mem) == 2

    def test_repr(self):
        mem = SummaryMemory()
        r = repr(mem)
        assert "SummaryMemory" in r


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    """Tests for WorkingMemory."""

    def test_store_and_retrieve(self):
        mem = WorkingMemory()
        mem.store("key1", "value1")
        assert mem.retrieve("key1") == "value1"

    def test_retrieve_nonexistent(self):
        mem = WorkingMemory()
        assert mem.retrieve("nonexistent") is None

    def test_retrieve_with_default(self):
        mem = WorkingMemory()
        assert mem.retrieve("missing", default="default") == "default"

    def test_delete(self):
        mem = WorkingMemory()
        mem.store("key1", "value1")
        assert mem.delete("key1") is True
        assert mem.delete("key1") is False

    def test_list_keys(self):
        mem = WorkingMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        keys = mem.list_keys()
        assert "a" in keys
        assert "b" in keys

    def test_clear(self):
        mem = WorkingMemory()
        mem.store("key1", "value1")
        mem.clear()
        assert len(mem) == 0

    def test_update(self):
        mem = WorkingMemory()
        mem.store("key1", "value1")
        mem.store("key1", "value2")
        assert mem.retrieve("key1") == "value2"

    def test_has_key(self):
        mem = WorkingMemory()
        mem.store("key1", "value1")
        assert mem.has_key("key1") is True
        assert mem.has_key("nonexistent") is False

    def test_contains(self):
        mem = WorkingMemory()
        mem.store("key1", "value1")
        assert "key1" in mem

    def test_as_dict(self):
        mem = WorkingMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        d = mem.as_dict()
        assert d == {"a": 1, "b": 2}

    def test_ttl_expiration(self):
        mem = WorkingMemory(default_ttl=0.01)
        mem.store("ephemeral", "data")
        import time
        time.sleep(0.05)
        assert mem.retrieve("ephemeral") is None

    def test_len(self):
        mem = WorkingMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        assert len(mem) == 2

    def test_repr(self):
        mem = WorkingMemory()
        r = repr(mem)
        assert "WorkingMemory" in r


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_init(self):
        mm = MemoryManager()
        assert mm is not None

    def test_create_conversation_memory(self):
        mm = MemoryManager()
        mem = mm.create_memory("conversation", {"max_messages": 50})
        assert isinstance(mem, ConversationMemory)

    def test_create_working_memory(self):
        mm = MemoryManager()
        mem = mm.create_memory("working")
        assert isinstance(mem, WorkingMemory)

    def test_create_summary_memory(self):
        mm = MemoryManager()
        mem = mm.create_memory("summary", {"summarize_threshold": 5})
        assert isinstance(mem, SummaryMemory)

    def test_invalid_memory_type(self):
        mm = MemoryManager()
        with pytest.raises(ValueError, match="Unknown memory type"):
            mm.create_memory("nonexistent")

    def test_get_memory(self):
        mm = MemoryManager()
        mem = mm.create_memory("conversation")
        retrieved = mm.get_memory(mem.id)
        assert retrieved is mem

    def test_get_nonexistent_memory(self):
        mm = MemoryManager()
        with pytest.raises(MemoryNotFoundError):
            mm.get_memory("nonexistent")

    def test_list_memories(self):
        mm = MemoryManager()
        mm.create_memory("conversation")
        mm.create_memory("working")
        ids = mm.list_memories()
        assert len(ids) == 2

    def test_remove_memory(self):
        mm = MemoryManager()
        mem = mm.create_memory("conversation")
        mm.remove_memory(mem.id)
        with pytest.raises(MemoryNotFoundError):
            mm.get_memory(mem.id)

    def test_clear(self):
        mm = MemoryManager()
        mm.create_memory("conversation")
        mm.create_memory("working")
        mm.clear()
        assert mm.list_memories() == []
