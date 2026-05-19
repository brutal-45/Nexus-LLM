"""
Nexus-LLM Chat History Module

Provides persistent chat history management with save/load sessions,
search functionality, and export/import capabilities.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


DEFAULT_HISTORY_DIR = os.path.expanduser("~/.nexus_llm/history")


@dataclass
class HistoryEntry:
    """A single entry in the chat history."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        """Deserialize from a dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id", ""),
            token_count=data.get("token_count", 0),
        )


@dataclass
class ChatSessionRecord:
    """Metadata about a saved chat session."""
    session_id: str
    created_at: float
    updated_at: float
    message_count: int
    model: str = ""
    title: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "model": self.model,
            "title": self.title,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatSessionRecord:
        """Deserialize from a dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            message_count=data.get("message_count", 0),
            model=data.get("model", ""),
            title=data.get("title", ""),
            tags=data.get("tags", []),
        )


class ChatHistory:
    """Manages chat history with persistence, search, and export.

    History entries are stored in memory and optionally persisted to
    JSON files on disk. Supports session management, full-text search,
    and multiple export formats.
    """

    def __init__(
        self,
        history_dir: str | None = None,
        max_entries: int = 10000,
        auto_persist: bool = True,
    ) -> None:
        self._entries: list[HistoryEntry] = []
        self._sessions: dict[str, ChatSessionRecord] = {}
        self._history_dir = Path(history_dir or DEFAULT_HISTORY_DIR)
        self._max_entries = max_entries
        self._auto_persist = auto_persist
        self._current_session_id = f"session_{int(time.time())}"

    @property
    def current_session_id(self) -> str:
        """Get the current session identifier."""
        return self._current_session_id

    @property
    def entry_count(self) -> int:
        """Get the total number of entries."""
        return len(self._entries)

    def add_entry(self, entry: HistoryEntry) -> None:
        """Add a new entry to the history.

        Args:
            entry: The HistoryEntry to add.
        """
        if not entry.session_id:
            entry.session_id = self._current_session_id
        entry.token_count = len(entry.content.split())
        self._entries.append(entry)
        self._enforce_max_entries()
        if self._auto_persist:
            self._persist_session()

    def _enforce_max_entries(self) -> None:
        """Enforce the maximum number of entries by removing oldest."""
        if len(self._entries) > self._max_entries:
            excess = len(self._entries) - self._max_entries
            self._entries = self._entries[excess:]

    def get_entries(
        self,
        session_id: str | None = None,
        role: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[HistoryEntry]:
        """Retrieve history entries with optional filtering.

        Args:
            session_id: Filter by session ID.
            role: Filter by role (user, assistant, system).
            limit: Maximum number of entries to return.
            offset: Number of entries to skip.

        Returns:
            Filtered list of HistoryEntry objects.
        """
        filtered = self._entries
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]
        if role:
            filtered = [e for e in filtered if e.role == role]
        filtered = filtered[offset:]
        if limit:
            filtered = filtered[:limit]
        return filtered

    def search(
        self,
        query: str,
        session_id: str | None = None,
        role: str | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> list[HistoryEntry]:
        """Search history entries by content.

        Args:
            query: Search query string or pattern.
            session_id: Optional session filter.
            role: Optional role filter.
            case_sensitive: Whether search is case-sensitive.
            regex: Whether to treat query as a regex pattern.

        Returns:
            Matching HistoryEntry objects.
        """
        entries = self.get_entries(session_id=session_id, role=role)
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(query, flags)
            return [e for e in entries if pattern.search(e.content)]

        if not case_sensitive:
            query_lower = query.lower()
            return [e for e in entries if query_lower in e.content.lower()]
        return [e for e in entries if query in e.content]

    def get_recent(self, count: int = 10, session_id: str | None = None) -> list[HistoryEntry]:
        """Get the most recent entries.

        Args:
            count: Number of recent entries to return.
            session_id: Optional session filter.

        Returns:
            List of the most recent HistoryEntry objects.
        """
        entries = self.get_entries(session_id=session_id)
        return entries[-count:]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the history.

        Returns:
            Dictionary with history statistics.
        """
        if not self._entries:
            return {
                "total_entries": 0,
                "sessions": 0,
                "total_tokens": 0,
                "roles": {},
            }
        role_counts: dict[str, int] = {}
        total_tokens = 0
        session_ids: set[str] = set()
        for entry in self._entries:
            role_counts[entry.role] = role_counts.get(entry.role, 0) + 1
            total_tokens += entry.token_count
            if entry.session_id:
                session_ids.add(entry.session_id)
        return {
            "total_entries": len(self._entries),
            "sessions": len(session_ids),
            "total_tokens": total_tokens,
            "roles": role_counts,
            "oldest": self._entries[0].timestamp,
            "newest": self._entries[-1].timestamp,
        }

    def save_session(self, session_id: str | None = None, path: str | None = None) -> str:
        """Save a session's history to a JSON file.

        Args:
            session_id: Session to save (defaults to current).
            path: Custom file path (defaults to history_dir/session_id.json).

        Returns:
            The path the session was saved to.
        """
        sid = session_id or self._current_session_id
        entries = self.get_entries(session_id=sid)
        if not entries:
            raise ValueError(f"No entries found for session: {sid}")

        session_data = {
            "session_id": sid,
            "saved_at": time.time(),
            "message_count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }

        save_path = Path(path) if path else self._history_dir / f"{sid}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        record = ChatSessionRecord(
            session_id=sid,
            created_at=entries[0].timestamp,
            updated_at=time.time(),
            message_count=len(entries),
        )
        self._sessions[sid] = record

        return str(save_path)

    def load_session(self, path: str) -> str:
        """Load a session from a JSON file.

        Args:
            path: Path to the session JSON file.

        Returns:
            The loaded session ID.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        session_id = data.get("session_id", f"loaded_{int(time.time())}")
        for entry_data in data.get("entries", []):
            entry = HistoryEntry.from_dict(entry_data)
            entry.session_id = session_id
            self._entries.append(entry)

        record = ChatSessionRecord(
            session_id=session_id,
            created_at=data.get("saved_at", time.time()),
            updated_at=time.time(),
            message_count=len(data.get("entries", [])),
        )
        self._sessions[session_id] = record

        return session_id

    def list_sessions(self) -> list[ChatSessionRecord]:
        """List all known sessions.

        Returns:
            List of ChatSessionRecord objects.
        """
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's history.

        Args:
            session_id: The session to delete.

        Returns:
            True if the session was found and deleted.
        """
        self._entries = [e for e in self._entries if e.session_id != session_id]
        if session_id in self._sessions:
            del self._sessions[session_id]
        # Also delete the file if it exists
        session_path = self._history_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            return True
        return session_id in self._sessions

    def export(
        self,
        format: str = "json",
        session_id: str | None = None,
        path: str | None = None,
    ) -> str:
        """Export history in various formats.

        Args:
            format: Export format - 'json', 'csv', 'markdown', 'text'.
            session_id: Optional session to export.
            path: Output file path.

        Returns:
            The exported content as a string, or the file path if path was given.
        """
        entries = self.get_entries(session_id=session_id)

        if format == "json":
            content = json.dumps(
                [e.to_dict() for e in entries],
                indent=2,
                ensure_ascii=False,
            )
        elif format == "csv":
            lines = ["role,content,timestamp,token_count"]
            for e in entries:
                escaped_content = e.content.replace('"', '""').replace("\n", "\\n")
                lines.append(f'{e.role},"{escaped_content}",{e.timestamp},{e.token_count}')
            content = "\n".join(lines)
        elif format == "markdown":
            lines = ["# Chat History", ""]
            if session_id:
                lines.append(f"Session: {session_id}", )
                lines.append("")
            for e in entries:
                lines.append(f"## {e.role.capitalize()}")
                lines.append(f"*{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e.timestamp))}*")
                lines.append("")
                lines.append(e.content)
                lines.append("")
                lines.append("---")
                lines.append("")
            content = "\n".join(lines)
        elif format == "text":
            lines = []
            for e in entries:
                ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
                lines.append(f"[{ts}] {e.role}: {e.content}")
                lines.append("")
            content = "\n".join(lines)
        else:
            raise ValueError(f"Unknown export format: {format}")

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return path

        return content

    def import_data(
        self,
        path: str,
        format: str = "json",
        session_id: str | None = None,
    ) -> int:
        """Import history from a file.

        Args:
            path: Path to the file to import.
            format: File format - 'json' or 'csv'.
            session_id: Optional session ID to assign.

        Returns:
            Number of entries imported.
        """
        sid = session_id or f"imported_{int(time.time())}"
        count = 0

        if format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else data.get("entries", [])
            for entry_data in entries:
                entry = HistoryEntry.from_dict(entry_data)
                entry.session_id = sid
                self._entries.append(entry)
                count += 1
        elif format == "csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = HistoryEntry(
                        role=row.get("role", "user"),
                        content=row.get("content", ""),
                        timestamp=float(row.get("timestamp", time.time())),
                        session_id=sid,
                        token_count=int(row.get("token_count", 0)),
                    )
                    self._entries.append(entry)
                    count += 1
        else:
            raise ValueError(f"Unsupported import format: {format}")

        return count

    def _persist_session(self) -> None:
        """Auto-persist the current session to disk."""
        try:
            self._history_dir.mkdir(parents=True, exist_ok=True)
            sid = self._current_session_id
            entries = self.get_entries(session_id=sid)
            if not entries:
                return
            session_path = self._history_dir / f"{sid}.json"
            session_data = {
                "session_id": sid,
                "updated_at": time.time(),
                "message_count": len(entries),
                "entries": [e.to_dict() for e in entries],
            }
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass  # Silently fail on persistence errors during auto-persist

    def clear(self, session_id: str | None = None) -> None:
        """Clear history entries.

        Args:
            session_id: If given, only clear entries for this session.
        """
        if session_id:
            self._entries = [e for e in self._entries if e.session_id != session_id]
            self._sessions.pop(session_id, None)
        else:
            self._entries.clear()
            self._sessions.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> HistoryEntry:
        return self._entries[index]
