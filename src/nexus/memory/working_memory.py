"""
Nexus Working Memory
=====================
Working memory (scratchpad) components for active reasoning and task tracking.

This module provides three key components for managing short-term, transient state:

1. **WorkingMemoryBuffer**: A limited-capacity key-value store that models the
   active working memory of an LLM agent. Entries are stored as key-value pairs
   and evicted using an LRU (Least Recently Used) policy when capacity is reached.

2. **Scratchpad**: A freeform text buffer for chain-of-thought reasoning,
   intermediate calculations, and scratch computations. It supports append,
   prepend, search, replace, and provides character/word counting.

3. **TaskStateTracker**: Tracks the state of multi-step tasks with key-value
   state management, progress tracking, and snapshot save/restore functionality.

Design Philosophy
-----------------
Working memory is volatile and capacity-limited, reflecting the cognitive constraint
that agents can only actively consider a small amount of information at once.
Unlike long-term memory, working memory is not persisted across sessions by default,
though it supports explicit snapshot/restore for checkpointing.

Typical Usage Patterns
----------------------
- **Reasoning trace**: Use Scratchpad to accumulate chain-of-thought reasoning steps.
- **Variable binding**: Use WorkingMemoryBuffer to store intermediate results keyed
  by meaningful names.
- **Task progress**: Use TaskStateTracker to monitor and report progress through
  multi-step tasks.
"""

import collections
import copy
import json
import os
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


# ═══════════════════════════════════════════════════════════════════════════════
# Working Memory Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class _LRUOrderedDict(collections.OrderedDict):
    """OrderedDict subclass with explicit LRU touch method."""

    def touch(self, key: str) -> None:
        """Move an existing key to the end (most recently used)."""
        if key in self:
            self.move_to_end(key)


class WorkingMemoryBuffer:
    """Limited-capacity key-value store with LRU eviction.

    This buffer models the active working memory of an LLM agent. It stores
    key-value pairs and automatically evicts the least recently used entries
    when capacity is reached.

    The buffer is thread-safe for concurrent read/write operations.

    Args:
        capacity: Maximum number of key-value pairs to store. When the capacity
            is exceeded, the least recently used entry is evicted.
        default_ttl: Optional default time-to-live in seconds for entries.
            If None, entries do not expire based on time.

    Example:
        >>> wm = WorkingMemoryBuffer(capacity=4)
        >>> wm.write("user_name", "Alice")
        >>> wm.write("task", "Summarize document")
        >>> wm.read("user_name")
        'Alice'
        >>> wm.keys()
        ['user_name', 'task']
        >>> wm.capacity()
        4
        >>> wm.is_full()
        False
    """

    def __init__(
        self,
        capacity: int = 64,
        default_ttl: Optional[float] = None,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._store: _LRUOrderedDict = _LRUOrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = collections.defaultdict(int)
        self._lock = threading.RLock()

        # Statistics
        self._total_writes = 0
        self._total_reads = 0
        self._total_updates = 0
        self._total_deletes = 0
        self._total_evictions = 0

    def write(self, key: str, value: Any) -> None:
        """Store a key-value pair in the working memory buffer.

        If the key already exists, it is updated (moved to the most-recently-used
        position). If the buffer is at capacity, the least recently used entry
        is evicted before the new entry is added.

        Args:
            key: String key for the entry. Must not be empty.
            value: Any serializable value to store.

        Raises:
            ValueError: If key is empty or None.
        """
        if not key:
            raise ValueError("key must be a non-empty string")

        with self._lock:
            # Check TTL and evict expired entries first
            self._evict_expired()

            # If key exists, update it (moves to end in OrderedDict)
            if key in self._store:
                del self._store[key]
                del self._timestamps[key]
                self._access_counts[key] += 1
            else:
                # Evict LRU if at capacity
                while len(self._store) >= self._capacity:
                    self._evict_lru()

            self._store[key] = value
            self._timestamps[key] = time.time()
            self._access_counts[key] += 1
            self._total_writes += 1

    def read(self, key: str, default: Any = None) -> Any:
        """Read the value associated with a key.

        Reading a key moves it to the most-recently-used position (LRU touch).

        Args:
            key: String key to look up.
            default: Value to return if key is not found. Defaults to None.

        Returns:
            The stored value, or the default if key is not found or has expired.
        """
        with self._lock:
            if key not in self._store:
                return default

            # Check TTL expiration
            if self._is_expired(key):
                del self._store[key]
                del self._timestamps[key]
                self._access_counts.pop(key, None)
                return default

            # Touch the key (move to end for LRU)
            self._store.touch(key)
            self._access_counts[key] += 1
            self._total_reads += 1

            return self._store[key]

    def update(self, key: str, value: Any) -> bool:
        """Update the value of an existing key.

        Unlike write(), this method only succeeds if the key already exists.
        The key is moved to the most-recently-used position.

        Args:
            key: String key to update.
            value: New value to associate with the key.

        Returns:
            True if the key was found and updated, False otherwise.
        """
        with self._lock:
            if key not in self._store:
                return False

            if self._is_expired(key):
                del self._store[key]
                del self._timestamps[key]
                self._access_counts.pop(key, None)
                return False

            self._store[key] = value
            self._store.touch(key)
            self._timestamps[key] = time.time()
            self._access_counts[key] += 1
            self._total_updates += 1

            return True

    def delete(self, key: str) -> bool:
        """Delete a key-value pair from the buffer.

        Args:
            key: String key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._timestamps.pop(key, None)
                self._access_counts.pop(key, None)
                self._total_deletes += 1
                return True
            return False

    def has(self, key: str) -> bool:
        """Check if a key exists in the buffer (and is not expired).

        This method does NOT touch the key (does not affect LRU order).

        Args:
            key: String key to check.

        Returns:
            True if the key exists and is not expired.
        """
        with self._lock:
            if key not in self._store:
                return False
            if self._is_expired(key):
                del self._store[key]
                del self._timestamps[key]
                self._access_counts.pop(key, None)
                return False
            return True

    def keys(self) -> List[str]:
        """Return a list of all keys in the buffer, ordered by LRU (oldest first).

        Expired entries are silently removed.

        Returns:
            List of key strings.
        """
        with self._lock:
            self._evict_expired()
            return list(self._store.keys())

    def values(self) -> List[Any]:
        """Return a list of all values in the buffer, ordered by LRU.

        Returns:
            List of stored values.
        """
        with self._lock:
            self._evict_expired()
            return list(self._store.values())

    def items(self) -> List[Tuple[str, Any]]:
        """Return a list of (key, value) tuples, ordered by LRU.

        Returns:
            List of (key, value) tuples.
        """
        with self._lock:
            self._evict_expired()
            return list(self._store.items())

    def clear(self) -> int:
        """Remove all entries from the buffer.

        Returns:
            Number of entries that were removed.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._timestamps.clear()
            self._access_counts.clear()
            return count

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current buffer state.

        A snapshot is a serializable dictionary containing all entries,
        timestamps, and access counts. It can be restored later using
        the restore() method.

        Returns:
            Dictionary containing:
            - entries: Dict of key-value pairs
            - timestamps: Dict of key-timestamp pairs
            - access_counts: Dict of key-access_count pairs
            - created_at: Snapshot creation timestamp
        """
        with self._lock:
            return {
                "entries": copy.deepcopy(dict(self._store)),
                "timestamps": dict(self._timestamps),
                "access_counts": dict(self._access_counts),
                "created_at": time.time(),
                "capacity": self._capacity,
            }

    def restore(self, snapshot: Dict[str, Any]) -> int:
        """Restore the buffer from a previously created snapshot.

        This replaces all current entries with those from the snapshot.
        Entries that don't fit within capacity are dropped (LRU order).

        Args:
            snapshot: Dictionary created by the snapshot() method.

        Returns:
            Number of entries restored.
        """
        with self._lock:
            entries = snapshot.get("entries", {})
            timestamps = snapshot.get("timestamps", {})
            access_counts = snapshot.get("access_counts", {})

            self._store.clear()
            self._timestamps.clear()
            self._access_counts.clear()

            # Restore in original order, respecting capacity
            restored = 0
            for key, value in entries.items():
                if restored >= self._capacity:
                    break
                self._store[key] = value
                if key in timestamps:
                    self._timestamps[key] = timestamps[key]
                else:
                    self._timestamps[key] = time.time()
                if key in access_counts:
                    self._access_counts[key] = access_counts[key]
                restored += 1

            return restored

    def capacity(self) -> int:
        """Return the maximum capacity of the buffer.

        Returns:
            Maximum number of entries the buffer can hold.
        """
        return self._capacity

    def is_full(self) -> bool:
        """Check if the buffer is at capacity.

        Note: expired entries are evicted before checking.

        Returns:
            True if the buffer contains the maximum number of entries.
        """
        with self._lock:
            self._evict_expired()
            return len(self._store) >= self._capacity

    def size(self) -> int:
        """Return the current number of entries (excluding expired).

        Returns:
            Current number of stored entries.
        """
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def remaining_capacity(self) -> int:
        """Return how many more entries can be stored.

        Returns:
            Number of available slots before eviction begins.
        """
        with self._lock:
            self._evict_expired()
            return max(0, self._capacity - len(self._store))

    def get_access_count(self, key: str) -> int:
        """Get the number of times a key has been accessed.

        Args:
            key: String key to query.

        Returns:
            Access count, or 0 if key does not exist.
        """
        with self._lock:
            return self._access_counts.get(key, 0)

    def get_age(self, key: str) -> float:
        """Get the age of an entry in seconds since it was written.

        Args:
            key: String key to query.

        Returns:
            Age in seconds, or -1.0 if key does not exist.
        """
        with self._lock:
            if key not in self._timestamps:
                return -1.0
            return time.time() - self._timestamps[key]

    def set_ttl(self, key: str, ttl_seconds: float) -> bool:
        """Set a time-to-live for a specific key.

        The entry will be automatically evicted after the specified TTL
        has elapsed since the last write.

        Args:
            key: String key to set TTL for.
            ttl_seconds: Time-to-live in seconds. Must be positive.

        Returns:
            True if the key exists and TTL was set, False otherwise.
        """
        if key not in self._store:
            return False
        # Store TTL as a special metadata entry
        self._timestamps[f"__ttl__{key}"] = ttl_seconds
        return True

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the working memory buffer.

        Returns:
            Dictionary with buffer statistics.
        """
        with self._lock:
            self._evict_expired()
            return {
                "size": len(self._store),
                "capacity": self._capacity,
                "utilization": len(self._store) / self._capacity if self._capacity > 0 else 0.0,
                "remaining": max(0, self._capacity - len(self._store)),
                "total_writes": self._total_writes,
                "total_reads": self._total_reads,
                "total_updates": self._total_updates,
                "total_deletes": self._total_deletes,
                "total_evictions": self._total_evictions,
                "has_ttl": self._default_ttl is not None,
                "default_ttl": self._default_ttl,
            }

    def export_json(self, path: str) -> None:
        """Export buffer state to a JSON file.

        Args:
            path: Filesystem path for the output file.
        """
        data = {
            "capacity": self._capacity,
            "default_ttl": self._default_ttl,
            "entries": {},
            "timestamps": self._timestamps,
            "access_counts": dict(self._access_counts),
            "exported_at": time.time(),
        }
        for key, value in self._store.items():
            try:
                json.dumps(value)
                data["entries"][key] = value
            except (TypeError, ValueError):
                data["entries"][key] = str(value)

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def import_json(self, path: str) -> int:
        """Import buffer state from a JSON file.

        Clears existing entries before importing.

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            Number of entries imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._store.clear()
        self._timestamps.clear()
        self._access_counts.clear()

        entries = data.get("entries", {})
        timestamps = data.get("timestamps", {})
        access_counts = data.get("access_counts", {})

        count = 0
        for key, value in entries.items():
            if count >= self._capacity:
                break
            self._store[key] = value
            if key in timestamps:
                self._timestamps[key] = timestamps[key]
            if key in access_counts:
                self._access_counts[key] = access_counts[key]
            count += 1

        return count

    def _evict_lru(self) -> Optional[str]:
        """Evict the least recently used entry.

        Returns:
            The key that was evicted, or None if buffer is empty.
        """
        if not self._store:
            return None

        # popitem(last=False) removes the first (oldest/LRU) entry
        key, _ = self._store.popitem(last=False)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
        self._total_evictions += 1
        return key

    def _evict_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries evicted.
        """
        if self._default_ttl is None:
            return 0

        now = time.time()
        expired_keys = []
        for key, ts in list(self._timestamps.items()):
            if key.startswith("__ttl__"):
                continue
            if now - ts > self._default_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            if key in self._store:
                del self._store[key]
                self._timestamps.pop(key, None)
                self._access_counts.pop(key, None)
                self._total_evictions += 1

        return len(expired_keys)

    def _is_expired(self, key: str) -> bool:
        """Check if a specific key has expired.

        Args:
            key: String key to check.

        Returns:
            True if the key has expired, False otherwise.
        """
        if self._default_ttl is None:
            return False

        # Check for per-key TTL
        ttl_key = f"__ttl__{key}"
        if ttl_key in self._timestamps:
            ttl = self._timestamps[ttl_key]
        else:
            ttl = self._default_ttl

        ts = self._timestamps.get(key)
        if ts is None:
            return True

        return (time.time() - ts) > ttl

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        return (
            f"WorkingMemoryBuffer(size={self.size()}/{self._capacity}, "
            f"evictions={self._total_evictions})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Scratchpad
# ═══════════════════════════════════════════════════════════════════════════════

class Scratchpad:
    """Freeform text buffer for chain-of-thought reasoning and scratch computation.

    The scratchpad provides a simple text area where an LLM agent can accumulate
    intermediate reasoning steps, perform scratch calculations, and build up
    responses incrementally. It supports text manipulation operations such as
    append, prepend, search, and replace.

    The scratchpad enforces a maximum character length to prevent unbounded growth.
    When content exceeds the limit, it is automatically truncated from the beginning.

    Args:
        max_length: Maximum character length of the scratchpad content.
            When exceeded, content is truncated from the beginning.
        separator: Default separator used when appending text. Defaults to "\\n".

    Example:
        >>> pad = Scratchpad(max_length=1000)
        >>> pad.append("Step 1: Read the problem")
        >>> pad.append("Step 2: Identify key variables")
        >>> pad.read()
        'Step 1: Read the problem\\nStep 2: Identify key variables'
        >>> pad.word_count()
        11
        >>> pad.search("key variables")
        'ep 2: Identify key variables'
    """

    def __init__(
        self,
        max_length: int = 4096,
        separator: str = "\n",
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self._max_length = max_length
        self._separator = separator
        self._content: str = ""
        self._lock = threading.RLock()

        # History tracking
        self._history: List[str] = []
        self._max_history: int = 50
        self._total_operations: int = 0

    def append(self, text: str) -> int:
        """Append text to the end of the scratchpad.

        The text is separated from existing content by the configured separator.
        If the resulting content exceeds max_length, it is truncated from the
        beginning.

        Args:
            text: Text to append.

        Returns:
            New total character count.
        """
        if not text:
            return self.char_count()

        with self._lock:
            self._save_history()
            if self._content:
                self._content += self._separator + text
            else:
                self._content = text
            self._enforce_max_length()
            self._total_operations += 1
            return self.char_count()

    def prepend(self, text: str) -> int:
        """Prepend text to the beginning of the scratchpad.

        If the resulting content exceeds max_length, it is truncated from the end.

        Args:
            text: Text to prepend.

        Returns:
            New total character count.
        """
        if not text:
            return self.char_count()

        with self._lock:
            self._save_history()
            if self._content:
                self._content = text + self._separator + self._content
            else:
                self._content = text
            self._enforce_max_length(truncate_from="end")
            self._total_operations += 1
            return self.char_count()

    def read(self) -> str:
        """Read the entire content of the scratchpad.

        Returns:
            The full text content of the scratchpad.
        """
        with self._lock:
            return self._content

    def read_last(self, n: int = 1) -> str:
        """Read the last N lines from the scratchpad.

        Lines are split by the configured separator (default: newline).

        Args:
            n: Number of lines to return from the end.

        Returns:
            The last N lines joined by newlines. If n exceeds the number of
            lines, all lines are returned.
        """
        with self._lock:
            if not self._content:
                return ""

            lines = self._content.split(self._separator)
            # Filter empty trailing lines
            while lines and not lines[-1]:
                lines.pop()

            last_n = lines[-n:] if n > 0 else lines
            return self._separator.join(last_n)

    def read_first(self, n: int = 1) -> str:
        """Read the first N lines from the scratchpad.

        Args:
            n: Number of lines to return from the beginning.

        Returns:
            The first N lines joined by newlines.
        """
        with self._lock:
            if not self._content:
                return ""

            lines = self._content.split(self._separator)
            first_n = lines[:n] if n > 0 else lines
            return self._separator.join(first_n)

    def read_range(self, start: int, end: int) -> str:
        """Read a range of characters from the scratchpad.

        Args:
            start: Starting character index (inclusive). Supports negative indexing.
            end: Ending character index (exclusive). Supports negative indexing.

        Returns:
            Substring of the content in the specified range.
        """
        with self._lock:
            return self._content[start:end]

    def clear(self) -> str:
        """Clear all content from the scratchpad.

        Returns:
            The content that was cleared.
        """
        with self._lock:
            self._save_history()
            cleared = self._content
            self._content = ""
            self._total_operations += 1
            return cleared

    def search(self, query: str, case_sensitive: bool = False) -> str:
        """Search for a text pattern in the scratchpad and return the context.

        Finds the first occurrence of the query and returns a surrounding
        context window of up to 200 characters.

        Args:
            query: Text pattern to search for.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            A context string containing the match and surrounding text.
            Returns an empty string if the query is not found.
        """
        if not query:
            return ""

        with self._lock:
            if not self._content:
                return ""

            search_content = self._content if case_sensitive else self._content.lower()
            search_query = query if case_sensitive else query.lower()

            index = search_content.find(search_query)
            if index == -1:
                return ""

            # Extract context window (200 chars around the match)
            context_size = 200
            start = max(0, index - context_size // 2)
            end = min(len(self._content), index + len(query) + context_size // 2)

            result = self._content[start:end]
            if start > 0:
                result = "..." + result
            if end < len(self._content):
                result = result + "..."

            return result

    def find_all(self, query: str, case_sensitive: bool = False) -> List[int]:
        """Find all occurrences of a query in the scratchpad.

        Args:
            query: Text pattern to search for.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            List of character indices where the query was found.
        """
        if not query:
            return []

        with self._lock:
            if not self._content:
                return []

            search_content = self._content if case_sensitive else self._content.lower()
            search_query = query if case_sensitive else query.lower()

            positions = []
            start = 0
            while True:
                index = search_content.find(search_query, start)
                if index == -1:
                    break
                positions.append(index)
                start = index + len(query)

            return positions

    def replace(self, old: str, new: str, count: int = -1) -> int:
        """Replace occurrences of text in the scratchpad.

        Args:
            old: Text to find and replace.
            new: Replacement text.
            count: Maximum number of replacements. If -1, replace all occurrences.

        Returns:
            Number of replacements made.
        """
        if not old or old == new:
            return 0

        with self._lock:
            self._save_history()
            if count == -1:
                self._content = self._content.replace(old, new)
                replacements = self._content.count(new)  # approximate
            else:
                self._content = self._content.replace(old, new, count)
                replacements = count

            self._total_operations += 1
            return replacements

    def replace_regex(self, pattern: str, replacement: str) -> int:
        """Replace text matching a regular expression pattern.

        Args:
            pattern: Regular expression pattern to match.
            replacement: Replacement string (can use group references).

        Returns:
            Number of replacements made.
        """
        import re

        if not pattern:
            return 0

        with self._lock:
            self._save_history()
            new_content, count = re.subn(pattern, replacement, self._content)
            self._content = new_content
            self._total_operations += 1
            return count

    def insert(self, text: str, position: int) -> int:
        """Insert text at a specific position.

        Args:
            text: Text to insert.
            position: Character position at which to insert. Supports negative indexing.

        Returns:
            New total character count.
        """
        with self._lock:
            self._save_history()
            self._content = self._content[:position] + text + self._content[position:]
            self._enforce_max_length()
            self._total_operations += 1
            return self.char_count()

    def delete_range(self, start: int, end: int) -> str:
        """Delete a range of characters from the scratchpad.

        Args:
            start: Starting character index (inclusive). Supports negative indexing.
            end: Ending character index (exclusive). Supports negative indexing.

        Returns:
            The deleted text.
        """
        with self._lock:
            self._save_history()
            deleted = self._content[start:end]
            self._content = self._content[:start] + self._content[end:]
            self._total_operations += 1
            return deleted

    def word_count(self) -> int:
        """Count the number of words in the scratchpad.

        Words are defined as sequences of non-whitespace characters.

        Returns:
            Number of words.
        """
        with self._lock:
            if not self._content.strip():
                return 0
            return len(self._content.split())

    def char_count(self) -> int:
        """Count the number of characters in the scratchpad.

        Returns:
            Number of characters.
        """
        with self._lock:
            return len(self._content)

    def line_count(self) -> int:
        """Count the number of lines in the scratchpad.

        Lines are defined as segments separated by the configured separator.

        Returns:
            Number of lines.
        """
        with self._lock:
            if not self._content:
                return 0
            return len(self._content.split(self._separator))

    def contains(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if the scratchpad contains a specific text.

        Args:
            text: Text to search for.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            True if the text is found.
        """
        if not text:
            return False
        with self._lock:
            if case_sensitive:
                return text in self._content
            return text.lower() in self._content.lower()

    def starts_with(self, text: str) -> bool:
        """Check if the scratchpad starts with the given text.

        Args:
            text: Text to check.

        Returns:
            True if the scratchpad starts with the text.
        """
        with self._lock:
            return self._content.startswith(text)

    def ends_with(self, text: str) -> bool:
        """Check if the scratchpad ends with the given text.

        Args:
            text: Text to check.

        Returns:
            True if the scratchpad ends with the text.
        """
        with self._lock:
            return self._content.endswith(text)

    def undo(self) -> bool:
        """Undo the last operation by restoring from history.

        Returns:
            True if undo was successful (history available), False otherwise.
        """
        with self._lock:
            if not self._history:
                return False
            self._content = self._history.pop()
            return True

    def set_content(self, text: str) -> int:
        """Replace the entire content of the scratchpad.

        Args:
            text: New content.

        Returns:
            New character count.
        """
        with self._lock:
            self._save_history()
            self._content = text
            self._enforce_max_length()
            self._total_operations += 1
            return self.char_count()

    def get_lines(self) -> List[str]:
        """Return all lines as a list.

        Returns:
            List of line strings.
        """
        with self._lock:
            if not self._content:
                return []
            return self._content.split(self._separator)

    def snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the scratchpad.

        Returns:
            Dictionary containing content and metadata.
        """
        with self._lock:
            return {
                "content": self._content,
                "max_length": self._max_length,
                "separator": self._separator,
                "word_count": self.word_count(),
                "char_count": self.char_count(),
                "line_count": self.line_count(),
                "created_at": time.time(),
            }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore the scratchpad from a snapshot.

        Args:
            snapshot: Dictionary created by snapshot().
        """
        with self._lock:
            self._content = snapshot.get("content", "")
            new_max = snapshot.get("max_length")
            if new_max is not None:
                self._max_length = new_max
            new_sep = snapshot.get("separator")
            if new_sep is not None:
                self._separator = new_sep
            self._enforce_max_length()

    def stats(self) -> Dict[str, Any]:
        """Return scratchpad statistics.

        Returns:
            Dictionary with usage statistics.
        """
        with self._lock:
            return {
                "char_count": len(self._content),
                "max_length": self._max_length,
                "utilization": len(self._content) / self._max_length if self._max_length > 0 else 0.0,
                "word_count": self.word_count(),
                "line_count": self.line_count(),
                "total_operations": self._total_operations,
                "history_size": len(self._history),
            }

    def export_json(self, path: str) -> None:
        """Export scratchpad content to a JSON file.

        Args:
            path: Filesystem path for the output file.
        """
        data = {
            "content": self._content,
            "max_length": self._max_length,
            "separator": self._separator,
            "exported_at": time.time(),
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_json(self, path: str) -> None:
        """Import scratchpad content from a JSON file.

        Args:
            path: Filesystem path to the JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            self._content = data.get("content", "")
            new_max = data.get("max_length")
            if new_max is not None:
                self._max_length = new_max
            new_sep = data.get("separator")
            if new_sep is not None:
                self._separator = new_sep
            self._enforce_max_length()

    def _enforce_max_length(self, truncate_from: str = "start") -> None:
        """Enforce the maximum length constraint.

        Args:
            truncate_from: 'start' to remove from the beginning,
                'end' to remove from the end.
        """
        if len(self._content) <= self._max_length:
            return

        if truncate_from == "start":
            # Try to truncate at a separator boundary
            self._content = self._content[-self._max_length:]
            # Find first separator to avoid partial lines
            sep_idx = self._content.find(self._separator)
            if sep_idx != -1 and sep_idx < len(self._content) // 2:
                self._content = self._content[sep_idx + len(self._separator):]
        elif truncate_from == "end":
            self._content = self._content[:self._max_length]
            # Find last separator to avoid partial lines
            sep_idx = self._content.rfind(self._separator)
            if sep_idx > len(self._content) // 2:
                self._content = self._content[:sep_idx]

    def _save_history(self) -> None:
        """Save current content to undo history."""
        self._history.append(self._content)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def __len__(self) -> int:
        return self.char_count()

    def __str__(self) -> str:
        return self._content

    def __repr__(self) -> str:
        preview = self._content[:50] + ("..." if len(self._content) > 50 else "")
        return f"Scratchpad(length={self.char_count()}, preview={preview!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# Task State Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class TaskStateTracker:
    """Tracks the state and progress of multi-step tasks.

    The TaskStateTracker maintains a key-value state dictionary that represents
    the current progress of a task. It provides methods for updating state,
    checking completion, computing progress percentage, and saving/loading
    state for checkpointing.

    The tracker is designed to be used in multi-step workflows where the agent
    needs to remember what has been done and what remains.

    Args:
        task_name: Human-readable name for the task being tracked.
        required_keys: Optional set of keys that must be present for the task
            to be considered complete.
        persistence_path: Optional path for auto-saving state.
        max_history: Maximum number of state change entries to keep in history.

    Example:
        >>> tracker = TaskStateTracker(
        ...     task_name="document_analysis",
        ...     required_keys={"file_read", "parsed", "summarized", "reviewed"},
        ... )
        >>> tracker.update("file_read", True)
        >>> tracker.update("parsed", True)
        >>> tracker.progress()
        0.5
        >>> tracker.is_complete()
        False
        >>> tracker.update("summarized", True)
        >>> tracker.update("reviewed", True)
        >>> tracker.is_complete()
        True
        >>> tracker.progress()
        1.0
    """

    def __init__(
        self,
        task_name: str = "unnamed_task",
        required_keys: Optional[Set[str]] = None,
        persistence_path: Optional[str] = None,
        max_history: int = 100,
    ):
        self._task_name = task_name
        self._required_keys = set(required_keys) if required_keys else set()
        self._persistence_path = persistence_path
        self._max_history = max_history

        self._state: Dict[str, Any] = {}
        self._created_at: float = time.time()
        self._modified_at: float = self._created_at
        self._completed: bool = False
        self._started: bool = False

        # History of state changes
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        # Step tracking
        self._steps: List[Dict[str, Any]] = []
        self._current_step: int = 0

        # Load persisted state if available
        if persistence_path:
            self.load()

    def update(self, key: str, value: Any) -> None:
        """Update a state key with a new value.

        Records the change in history and updates the modification timestamp.
        If the task has required keys, checks for completion after each update.

        Args:
            key: State key to update.
            value: New value for the key.
        """
        if not key:
            raise ValueError("key must be a non-empty string")

        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            self._started = True
            self._modified_at = time.time()

            # Record history
            history_entry = {
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "timestamp": self._modified_at,
                "step": self._current_step,
            }
            self._history.append(history_entry)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # Check completion
            self._check_completion()

            # Auto-save
            if self._persistence_path:
                self.save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of a state key.

        Args:
            key: State key to retrieve.
            default: Default value if key is not found.

        Returns:
            The stored value, or the default.
        """
        with self._lock:
            return self._state.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Return a copy of the complete state dictionary.

        Returns:
            Dictionary containing all state key-value pairs.
        """
        with self._lock:
            return copy.deepcopy(self._state)

    def has(self, key: str) -> bool:
        """Check if a state key exists.

        Args:
            key: State key to check.

        Returns:
            True if the key exists in the state.
        """
        with self._lock:
            return key in self._state

    def remove(self, key: str) -> bool:
        """Remove a state key.

        Args:
            key: State key to remove.

        Returns:
            True if the key was found and removed.
        """
        with self._lock:
            if key in self._state:
                old_value = self._state.pop(key)
                self._history.append({
                    "key": key,
                    "old_value": old_value,
                    "new_value": None,
                    "timestamp": time.time(),
                    "removed": True,
                })
                self._modified_at = time.time()
                self._check_completion()
                return True
            return False

    def is_complete(self) -> bool:
        """Check if the task is complete.

        A task is complete if all required keys have non-None values that
        evaluate to True in a boolean context, OR if the completed flag
        has been explicitly set.

        Returns:
            True if the task is complete.
        """
        with self._lock:
            if self._completed:
                return True
            if not self._required_keys:
                return self._completed
            return all(
                key in self._state and bool(self._state[key]) is True
                for key in self._required_keys
            )

    def set_complete(self, complete: bool = True) -> None:
        """Explicitly set or clear the completion status.

        Args:
            complete: True to mark as complete, False to mark as incomplete.
        """
        with self._lock:
            self._completed = complete
            self._modified_at = time.time()

    def progress(self) -> float:
        """Compute the progress of the task as a fraction.

        If required_keys are defined, progress is the fraction of required
        keys that have been set to truthy values.

        If no required_keys are defined and steps have been added,
        progress is the fraction of steps completed.

        If neither is defined, returns 1.0 if the task has been started,
        or 0.0 if not.

        Returns:
            Progress fraction in [0.0, 1.0].
        """
        with self._lock:
            if self._completed:
                return 1.0

            if self._required_keys:
                completed = sum(
                    1 for key in self._required_keys
                    if key in self._state and bool(self._state[key])
                )
                return completed / len(self._required_keys)

            if self._steps:
                completed = sum(
                    1 for step in self._steps if step.get("completed", False)
                )
                return completed / len(self._steps)

            return 1.0 if self._started else 0.0

    def reset(self) -> Dict[str, Any]:
        """Reset the task state to its initial state.

        Clears all state, history, and completion status while preserving
        configuration (task name, required keys, persistence path).

        Returns:
            The state dictionary that was cleared.
        """
        with self._lock:
            old_state = copy.deepcopy(self._state)
            self._state.clear()
            self._history.clear()
            self._steps.clear()
            self._completed = False
            self._started = False
            self._current_step = 0
            self._modified_at = time.time()
            self._created_at = time.time()

            if self._persistence_path:
                self.save()

            return old_state

    def save(self, path: Optional[str] = None) -> bool:
        """Save the current task state to a JSON file.

        Args:
            path: Filesystem path for the output file. If None, uses
                the configured persistence_path.

        Returns:
            True if save was successful.
        """
        save_path = path or self._persistence_path
        if not save_path:
            return False

        try:
            data = {
                "task_name": self._task_name,
                "required_keys": list(self._required_keys),
                "state": self._state,
                "completed": self._completed,
                "started": self._started,
                "current_step": self._current_step,
                "steps": self._steps,
                "history": self._history[-20:],  # Save last 20 history entries
                "created_at": self._created_at,
                "modified_at": self._modified_at,
                "saved_at": time.time(),
            }

            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            return True
        except (IOError, OSError, TypeError):
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """Load task state from a JSON file.

        Args:
            path: Filesystem path to the JSON file. If None, uses
                the configured persistence_path.

        Returns:
            True if load was successful.
        """
        load_path = path or self._persistence_path
        if not load_path or not os.path.exists(load_path):
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                self._task_name = data.get("task_name", self._task_name)
                self._required_keys = set(data.get("required_keys", []))
                self._state = data.get("state", {})
                self._completed = data.get("completed", False)
                self._started = data.get("started", False)
                self._current_step = data.get("current_step", 0)
                self._steps = data.get("steps", [])
                self._history = data.get("history", [])
                self._created_at = data.get("created_at", time.time())
                self._modified_at = data.get("modified_at", time.time())

            return True
        except (IOError, OSError, json.JSONDecodeError):
            return False

    def add_step(self, name: str, description: str = "") -> int:
        """Add a step to the task workflow.

        Steps provide a structured way to track task progress. Each step
        can be marked as completed independently.

        Args:
            name: Name of the step.
            description: Optional description of what the step does.

        Returns:
            Step index (0-based).
        """
        with self._lock:
            step = {
                "name": name,
                "description": description,
                "completed": False,
                "started_at": None,
                "completed_at": None,
            }
            self._steps.append(step)
            return len(self._steps) - 1

    def complete_step(self, step_index: int) -> bool:
        """Mark a step as completed.

        Args:
            step_index: Index of the step to complete.

        Returns:
            True if the step was found and marked complete.
        """
        with self._lock:
            if 0 <= step_index < len(self._steps):
                step = self._steps[step_index]
                if not step["completed"]:
                    step["completed"] = True
                    step["completed_at"] = time.time()
                    if step["started_at"] is None:
                        step["started_at"] = step["completed_at"]
                    self._current_step = step_index + 1
                    self._modified_at = time.time()
                    self._check_completion()
                return True
            return False

    def start_step(self, step_index: int) -> bool:
        """Mark a step as started.

        Args:
            step_index: Index of the step.

        Returns:
            True if the step was found.
        """
        with self._lock:
            if 0 <= step_index < len(self._steps):
                step = self._steps[step_index]
                if step["started_at"] is None:
                    step["started_at"] = time.time()
                    self._current_step = step_index
                    self._modified_at = time.time()
                return True
            return False

    def get_steps(self) -> List[Dict[str, Any]]:
        """Return a copy of all steps and their statuses.

        Returns:
            List of step dictionaries with completion status.
        """
        with self._lock:
            return copy.deepcopy(self._steps)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the state change history.

        Args:
            limit: Maximum number of history entries to return.
                If None, returns all entries.

        Returns:
            List of history entries, most recent first.
        """
        with self._lock:
            history = list(reversed(self._history))
            if limit is not None:
                history = history[:limit]
            return history

    def get_missing_keys(self) -> List[str]:
        """Return the list of required keys that have not been set.

        Returns:
            List of required key names that are missing or falsy.
        """
        with self._lock:
            missing = []
            for key in self._required_keys:
                if key not in self._state or not bool(self._state[key]):
                    missing.append(key)
            return missing

    def merge_state(self, new_state: Dict[str, Any]) -> None:
        """Merge a dictionary into the current state.

        All key-value pairs from new_state are written to the current state.

        Args:
            new_state: Dictionary of state updates.
        """
        with self._lock:
            for key, value in new_state.items():
                old_value = self._state.get(key)
                self._state[key] = value
                self._history.append({
                    "key": key,
                    "old_value": old_value,
                    "new_value": value,
                    "timestamp": time.time(),
                })

            self._started = True
            self._modified_at = time.time()
            self._check_completion()

            if self._persistence_path:
                self.save()

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current task state.

        Returns:
            Dictionary containing all task state data.
        """
        with self._lock:
            return {
                "task_name": self._task_name,
                "required_keys": list(self._required_keys),
                "state": copy.deepcopy(self._state),
                "completed": self._completed,
                "started": self._started,
                "progress": self.progress(),
                "current_step": self._current_step,
                "steps": copy.deepcopy(self._steps),
                "missing_keys": self.get_missing_keys(),
                "created_at": self._created_at,
                "modified_at": self._modified_at,
            }

    def elapsed_time(self) -> float:
        """Return elapsed time since task creation in seconds.

        Returns:
            Elapsed time in seconds.
        """
        return time.time() - self._created_at

    def idle_time(self) -> float:
        """Return time since last modification in seconds.

        Returns:
            Idle time in seconds.
        """
        return time.time() - self._modified_at

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive task statistics.

        Returns:
            Dictionary with task statistics.
        """
        with self._lock:
            return {
                "task_name": self._task_name,
                "completed": self._completed,
                "started": self._started,
                "progress": self.progress(),
                "state_keys": len(self._state),
                "required_keys": list(self._required_keys),
                "missing_keys": self.get_missing_keys(),
                "total_steps": len(self._steps),
                "completed_steps": sum(1 for s in self._steps if s.get("completed")),
                "current_step": self._current_step,
                "history_entries": len(self._history),
                "elapsed_seconds": self.elapsed_time(),
                "idle_seconds": self.idle_time(),
            }

    def _check_completion(self) -> None:
        """Check if all required keys are satisfied and update completion status."""
        if self._required_keys:
            self._completed = all(
                key in self._state and bool(self._state[key])
                for key in self._required_keys
            )

    def __repr__(self) -> str:
        return (
            f"TaskStateTracker(name={self._task_name!r}, "
            f"progress={self.progress():.1%}, "
            f"complete={self._completed})"
        )
