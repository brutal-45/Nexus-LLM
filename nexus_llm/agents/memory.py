"""Agent memory systems: short-term, long-term, and episodic memory.

Provides multi-layered memory with different retention policies,
forgetting curves, and retrieval capabilities for agents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single item stored in memory."""

    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    ttl: Optional[float] = None  # Time-to-live in seconds

    @property
    def age(self) -> float:
        """Age of the memory item in seconds."""
        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if the item has expired based on TTL."""
        if self.ttl is None:
            return False
        return self.age > self.ttl

    @property
    def retention_score(self) -> float:
        """Compute a retention score based on Ebbinghaus forgetting curve."""
        if self.importance >= 1.0:
            return 1.0
        # Ebbinghaus forgetting curve: R = e^(-t/S)
        # S is stability, affected by importance and access count
        stability = max(1.0, self.importance * 100 * (1 + math.log1p(self.access_count)))
        retention = math.exp(-self.age / stability)
        return retention

    def access(self) -> Any:
        """Access the memory item, updating access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "importance": self.importance,
            "tags": self.tags,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        """Deserialize from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            timestamp=data.get("timestamp", time.time()),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            ttl=data.get("ttl"),
        )


class AgentMemory(ABC):
    """Abstract base class for agent memory systems."""

    @abstractmethod
    def store(self, key: str, value: Any, **kwargs) -> None:
        """Store a value in memory."""
        ...

    @abstractmethod
    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory."""
        ...

    @abstractmethod
    def forget(self, key: str) -> bool:
        """Remove a value from memory."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search memory for relevant items."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return number of items in memory."""
        ...


class ShortTermMemory(AgentMemory):
    """Short-term memory with limited capacity and TTL.

    Stores recent information with an LRU (Least Recently Used)
    eviction policy and optional time-to-live expiration.
    """

    def __init__(self, capacity: int = 100, default_ttl: Optional[float] = 3600.0):
        """Initialize short-term memory.

        Args:
            capacity: Maximum number of items to store.
            default_ttl: Default time-to-live in seconds (None = no expiry).
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()

    def store(self, key: str, value: Any, importance: float = 0.5, tags: Optional[List[str]] = None, ttl: Optional[float] = None, **kwargs) -> None:
        """Store a value in short-term memory."""
        # Evict if at capacity
        while len(self._items) >= self.capacity:
            self._evict_one()

        item = MemoryItem(
            key=key,
            value=value,
            importance=importance,
            tags=tags or [],
            ttl=ttl if ttl is not None else self.default_ttl,
        )
        self._items[key] = item
        self._items.move_to_end(key)  # Mark as recently used

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from short-term memory."""
        item = self._items.get(key)
        if item is None:
            return default

        if item.is_expired:
            del self._items[key]
            return default

        self._items.move_to_end(key)  # Update LRU position
        return item.access()

    def forget(self, key: str) -> bool:
        """Remove a value from short-term memory."""
        if key in self._items:
            del self._items[key]
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search short-term memory by key/tag similarity."""
        query_lower = query.lower()
        results: List[Tuple[str, Any, float]] = []

        for key, item in self._items.items():
            if item.is_expired:
                continue

            # Simple matching: key containment + tag matching
            score = 0.0
            if query_lower in key.lower():
                score += 0.8
            for tag in item.tags:
                if query_lower in tag.lower():
                    score += 0.3
            if score > 0:
                results.append((key, item.value, min(score, 1.0)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._items.clear()

    def count(self) -> int:
        """Return number of non-expired items."""
        self._cleanup_expired()
        return len(self._items)

    def _evict_one(self) -> None:
        """Evict the least valuable item (LRU with retention score)."""
        if not self._items:
            return

        # First try to remove expired items
        for key, item in self._items.items():
            if item.is_expired:
                del self._items[key]
                return

        # Evict lowest retention score item
        worst_key = min(self._items.keys(), key=lambda k: self._items[k].retention_score)
        del self._items[worst_key]

    def _cleanup_expired(self) -> None:
        """Remove all expired items."""
        expired = [k for k, v in self._items.items() if v.is_expired]
        for key in expired:
            del self._items[key]


class LongTermMemory(AgentMemory):
    """Long-term memory with persistence and semantic retrieval.

    Stores information persistently with higher capacity and
    importance-weighted retention. Supports disk persistence.
    """

    def __init__(self, capacity: int = 10000, persist_path: Optional[str] = None):
        self.capacity = capacity
        self.persist_path = Path(persist_path) if persist_path else None
        self._items: Dict[str, MemoryItem] = {}

        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def store(self, key: str, value: Any, importance: float = 0.5, tags: Optional[List[str]] = None, ttl: Optional[float] = None, **kwargs) -> None:
        """Store a value in long-term memory."""
        if len(self._items) >= self.capacity:
            self._evict_lowest_importance()

        item = MemoryItem(
            key=key,
            value=value,
            importance=importance,
            tags=tags or [],
            ttl=None,  # Long-term memory doesn't expire by default
        )
        self._items[key] = item

        if self.persist_path:
            self._save_to_disk()

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from long-term memory."""
        item = self._items.get(key)
        if item is None:
            return default
        return item.access()

    def forget(self, key: str) -> bool:
        """Remove a value from long-term memory."""
        if key in self._items:
            del self._items[key]
            if self.persist_path:
                self._save_to_disk()
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search long-term memory by key, tag, and value content."""
        query_lower = query.lower()
        results: List[Tuple[str, Any, float]] = []

        for key, item in self._items.items():
            score = 0.0

            # Key matching
            if query_lower in key.lower():
                score += 0.7

            # Tag matching
            for tag in item.tags:
                if query_lower in tag.lower():
                    score += 0.3

            # Value content matching (for string values)
            if isinstance(item.value, str) and query_lower in item.value.lower():
                score += 0.5

            # Boost by importance and access frequency
            score *= (0.5 + 0.5 * item.importance)
            if item.access_count > 0:
                score *= min(1.0 + 0.1 * math.log1p(item.access_count), 2.0)

            if score > 0:
                results.append((key, item.value, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all long-term memory."""
        self._items.clear()
        if self.persist_path:
            self._save_to_disk()

    def count(self) -> int:
        """Return number of items in long-term memory."""
        return len(self._items)

    def _evict_lowest_importance(self) -> None:
        """Evict the item with the lowest importance score."""
        if not self._items:
            return
        worst_key = min(self._items.keys(), key=lambda k: self._items[k].importance)
        del self._items[worst_key]

    def _save_to_disk(self) -> None:
        """Persist memory to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: item.to_dict() for key, item in self._items.items()}
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._items = {key: MemoryItem.from_dict(item_data) for key, item_data in data.items()}
            logger.info("Loaded %d items from long-term memory.", len(self._items))
        except Exception as e:
            logger.error("Failed to load long-term memory: %s", e)


@dataclass
class Episode:
    """A recorded episode (experience) in episodic memory."""

    episode_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    event: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    timestamp: float = field(default_factory=time.time)
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "event": self.event,
            "context": self.context,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "emotional_valence": self.emotional_valence,
            "tags": self.tags,
        }


class EpisodicMemory(AgentMemory):
    """Episodic memory for recording and recalling experiences.

    Stores discrete episodes with temporal context and emotional
    valence. Supports recall by similarity, recency, and
    emotional association.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._episodes: List[Episode] = []

    def store(self, key: str, value: Any, **kwargs) -> None:
        """Store an episode (key=event, value=outcome)."""
        episode = Episode(
            event=key,
            outcome=str(value) if value is not None else "",
            context=kwargs.get("context", {}),
            emotional_valence=kwargs.get("emotional_valence", 0.0),
            tags=kwargs.get("tags", []),
        )
        self._episodes.append(episode)

        # Evict oldest if over capacity
        while len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def record_episode(
        self,
        event: str,
        outcome: str = "",
        context: Optional[Dict[str, Any]] = None,
        emotional_valence: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> Episode:
        """Record a new episode.

        Args:
            event: Description of what happened.
            outcome: What resulted from the event.
            context: Additional context information.
            emotional_valence: Emotional association (-1.0 to 1.0).
            tags: Tags for categorization.

        Returns:
            The recorded Episode.
        """
        episode = Episode(
            event=event,
            outcome=outcome,
            context=context or {},
            emotional_valence=emotional_valence,
            tags=tags or [],
        )
        self._episodes.append(episode)

        while len(self._episodes) > self.capacity:
            self._episodes.pop(0)

        return episode

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve the most recent episode matching the event description."""
        for episode in reversed(self._episodes):
            if key.lower() in episode.event.lower():
                return episode.outcome
        return default

    def forget(self, key: str) -> bool:
        """Remove episodes matching the key."""
        original_len = len(self._episodes)
        self._episodes = [ep for ep in self._episodes if key.lower() not in ep.event.lower()]
        return len(self._episodes) < original_len

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search episodic memory for relevant episodes."""
        query_lower = query.lower()
        scored_episodes: List[Tuple[Episode, float]] = []

        for episode in self._episodes:
            score = 0.0

            # Event text matching
            if query_lower in episode.event.lower():
                score += 0.6

            # Outcome matching
            if query_lower in episode.outcome.lower():
                score += 0.3

            # Tag matching
            for tag in episode.tags:
                if query_lower in tag.lower():
                    score += 0.2

            # Recency boost (exponential decay)
            age_hours = (time.time() - episode.timestamp) / 3600
            recency_factor = math.exp(-0.1 * age_hours)
            score *= (0.5 + 0.5 * recency_factor)

            # Emotional salience boost
            score *= (1.0 + 0.3 * abs(episode.emotional_valence))

            if score > 0:
                scored_episodes.append((episode, score))

        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [
            (ep.event, ep.outcome, score)
            for ep, score in scored_episodes[:top_k]
        ]

    def recall_recent(self, n: int = 5) -> List[Episode]:
        """Recall the n most recent episodes."""
        return self._episodes[-n:]

    def recall_by_emotion(self, valence_range: Tuple[float, float] = (-1.0, 0.0), top_k: int = 5) -> List[Episode]:
        """Recall episodes within a range of emotional valence."""
        filtered = [
            ep for ep in self._episodes
            if valence_range[0] <= ep.emotional_valence <= valence_range[1]
        ]
        filtered.sort(key=lambda x: abs(x.emotional_valence), reverse=True)
        return filtered[:top_k]

    def clear(self) -> None:
        """Clear all episodic memory."""
        self._episodes.clear()

    def count(self) -> int:
        """Return number of episodes."""
        return len(self._episodes)
