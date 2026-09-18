"""Test agent memory for Nexus-LLM."""
import time
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class MemoryEntry:
    key: str
    value: Any
    timestamp: float = 0.0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class AgentMemory:
    def __init__(self, max_entries: int = 1000):
        self._max_entries = max_entries
        self._entries: Dict[str, MemoryEntry] = {}
        self._access_log: List[str] = []

    def store(self, key: str, value: Any, importance: float = 0.5, metadata: Dict = None):
        if key in self._entries:
            old = self._entries[key]
            self._entries[key] = MemoryEntry(key=key, value=value, importance=importance, metadata=metadata or old.metadata)
        else:
            if len(self._entries) >= self._max_entries:
                self._evict()
            self._entries[key] = MemoryEntry(key=key, value=value, importance=importance, metadata=metadata or {})

    def retrieve(self, key: str) -> Optional[Any]:
        if key not in self._entries:
            return None
        self._access_log.append(key)
        return self._entries[key].value

    def delete(self, key: str) -> bool:
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def search(self, query: str) -> List[MemoryEntry]:
        results = []
        query_lower = query.lower()
        for entry in self._entries.values():
            if query_lower in str(entry.value).lower() or query_lower in entry.key.lower():
                results.append(entry)
        results.sort(key=lambda e: e.importance, reverse=True)
        return results

    def get_important(self, min_importance: float = 0.7) -> List[MemoryEntry]:
        return [e for e in self._entries.values() if e.importance >= min_importance]

    def get_recent(self, count: int = 10) -> List[MemoryEntry]:
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.timestamp, reverse=True)
        return sorted_entries[:count]

    def clear(self):
        self._entries.clear()
        self._access_log.clear()

    def _evict(self):
        least_important = min(self._entries.values(), key=lambda e: e.importance)
        del self._entries[least_important.key]

    @property
    def size(self):
        return len(self._entries)

    @property
    def access_count(self):
        return len(self._access_log)


class TestMemoryEntry:
    def test_creation(self):
        entry = MemoryEntry(key="test", value="hello")
        assert entry.key == "test"
        assert entry.value == "hello"
        assert entry.importance == 0.5
        assert entry.timestamp > 0

    def test_custom_importance(self):
        entry = MemoryEntry(key="test", value="hello", importance=0.9)
        assert entry.importance == 0.9


class TestAgentMemory:
    def test_store_and_retrieve(self):
        memory = AgentMemory()
        memory.store("name", "Alice")
        assert memory.retrieve("name") == "Alice"

    def test_retrieve_nonexistent(self):
        memory = AgentMemory()
        assert memory.retrieve("nonexistent") is None

    def test_update(self):
        memory = AgentMemory()
        memory.store("key", "value1")
        memory.store("key", "value2")
        assert memory.retrieve("key") == "value2"

    def test_delete(self):
        memory = AgentMemory()
        memory.store("key", "value")
        assert memory.delete("key") is True
        assert memory.retrieve("key") is None

    def test_delete_nonexistent(self):
        memory = AgentMemory()
        assert memory.delete("nonexistent") is False

    def test_search(self):
        memory = AgentMemory()
        memory.store("topic_ml", "Machine learning info")
        memory.store("topic_nlp", "Natural language processing")
        memory.store("recipe", "Pasta recipe")
        results = memory.search("machine")
        assert len(results) >= 1

    def test_search_by_key(self):
        memory = AgentMemory()
        memory.store("user_name", "Alice")
        results = memory.search("user")
        assert len(results) >= 1

    def test_get_important(self):
        memory = AgentMemory()
        memory.store("important", "critical info", importance=0.9)
        memory.store("trivial", "minor info", importance=0.2)
        important = memory.get_important(min_importance=0.7)
        assert len(important) == 1
        assert important[0].key == "important"

    def test_get_recent(self):
        memory = AgentMemory()
        memory.store("first", "1")
        time.sleep(0.01)
        memory.store("second", "2")
        recent = memory.get_recent(count=1)
        assert len(recent) == 1
        assert recent[0].key == "second"

    def test_size(self):
        memory = AgentMemory()
        memory.store("a", 1)
        memory.store("b", 2)
        assert memory.size == 2

    def test_access_count(self):
        memory = AgentMemory()
        memory.store("key", "value")
        memory.retrieve("key")
        memory.retrieve("key")
        assert memory.access_count == 2

    def test_clear(self):
        memory = AgentMemory()
        memory.store("key", "value")
        memory.clear()
        assert memory.size == 0
        assert memory.access_count == 0

    def test_eviction(self):
        memory = AgentMemory(max_entries=2)
        memory.store("a", 1, importance=0.5)
        memory.store("b", 2, importance=0.1)
        memory.store("c", 3, importance=0.8)
        assert memory.size <= 2
        assert memory.retrieve("c") is not None
