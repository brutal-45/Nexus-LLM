"""Chat history manager for Nexus-LLM.

Saves, loads, searches, and exports conversation history as JSON files.
Supports auto-trimming to a configurable maximum number of entries.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class Message:
    """A single chat message."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """A named conversation consisting of messages."""
    id: str
    title: str = "Untitled"
    model: str = "unknown"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            title=data.get("title", "Untitled"),
            model=data.get("model", "unknown"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=messages,
        )

    def clear(self) -> None:
        """Remove all messages from this conversation."""
        self.messages.clear()
        self.updated_at = time.time()


class ChatHistory:
    """Manages chat history with save, load, search, and export capabilities.

    Conversations are persisted as individual JSON files inside a dedicated
    history directory.  The manager also auto-trims old conversations when
    the total exceeds *max_history*.
    """

    def __init__(self, history_dir: str = ".nexus_history", max_history: int = 1000) -> None:
        self.history_dir = Path(history_dir)
        self.max_history = max_history
        self._current: Optional[Conversation] = None
        self._ensure_dir()

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create the history directory if it doesn't exist."""
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Current conversation
    # ------------------------------------------------------------------

    @property
    def current(self) -> Optional[Conversation]:
        """The active conversation."""
        return self._current

    def new_conversation(self, title: str = "Untitled", model: str = "unknown") -> Conversation:
        """Start a new conversation and set it as current."""
        conv_id = f"conv_{int(time.time() * 1000)}"
        self._current = Conversation(id=conv_id, title=title, model=model)
        return self._current

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Message]:
        """Add a message to the current conversation."""
        if self._current is None:
            self.new_conversation()
        return self._current.add_message(role, content, metadata)

    def get_messages(self) -> List[Message]:
        """Return messages from the current conversation."""
        if self._current is None:
            return []
        return list(self._current.messages)

    def clear_current(self) -> None:
        """Clear all messages in the current conversation."""
        if self._current is not None:
            self._current.clear()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, conversation: Optional[Conversation] = None, filename: Optional[str] = None) -> str:
        """Save a conversation to a JSON file.

        Args:
            conversation: The conversation to save (defaults to current).
            filename: Custom filename (defaults to ``<id>.json``).

        Returns:
            The path of the saved file.
        """
        conv = conversation or self._current
        if conv is None:
            raise ValueError("No conversation to save")

        fname = filename or f"{conv.id}.json"
        filepath = self.history_dir / fname

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load(self, filename: str) -> Conversation:
        """Load a conversation from a JSON file and set it as current.

        Args:
            filename: Filename (or path) relative to the history directory.

        Returns:
            The loaded Conversation.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        filepath = self.history_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._current = Conversation.from_dict(data)
        return self._current

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List summary information for all saved conversations."""
        results: List[Dict[str, Any]] = []
        for filepath in sorted(self.history_dir.glob("*.json")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "filename": filepath.name,
                    "id": data.get("id", "unknown"),
                    "title": data.get("title", "Untitled"),
                    "model": data.get("model", "unknown"),
                    "message_count": len(data.get("messages", [])),
                    "updated_at": data.get("updated_at", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def delete(self, filename: str) -> bool:
        """Delete a saved conversation file.

        Returns:
            True if the file was deleted, False if it didn't exist.
        """
        filepath = self.history_dir / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search through all saved conversations for a text query.

        Searches message content and conversation titles.  Returns a list
        of match descriptors.
        """
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []

        for filepath in sorted(self.history_dir.glob("*.json")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue

            # Search title
            if query_lower in data.get("title", "").lower():
                results.append({
                    "filename": filepath.name,
                    "id": data.get("id", "unknown"),
                    "title": data.get("title", "Untitled"),
                    "match": "title",
                })
                if len(results) >= limit:
                    break

            # Search messages
            for msg in data.get("messages", []):
                if query_lower in msg.get("content", "").lower():
                    results.append({
                        "filename": filepath.name,
                        "id": data.get("id", "unknown"),
                        "title": data.get("title", "Untitled"),
                        "match": "message",
                        "role": msg.get("role", ""),
                        "snippet": msg.get("content", "")[:120],
                    })
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_markdown(self, conversation: Optional[Conversation] = None, filepath: Optional[str] = None) -> str:
        """Export a conversation as a Markdown file.

        Returns:
            The path of the exported file.
        """
        conv = conversation or self._current
        if conv is None:
            raise ValueError("No conversation to export")

        lines: List[str] = [
            f"# {conv.title}",
            f"**Model:** {conv.model}  ",
            f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conv.created_at))}  ",
            f"**Messages:** {len(conv.messages)}  ",
            "",
            "---",
            "",
        ]

        for msg in conv.messages:
            role_label = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System",
            }.get(msg.role, msg.role.title())
            lines.append(f"## {role_label}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        out_path = filepath or str(self.history_dir / f"{conv.id}.md")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return out_path

    def export_json(self, conversation: Optional[Conversation] = None, filepath: Optional[str] = None) -> str:
        """Export a conversation as a standalone JSON file.

        Returns:
            The path of the exported file.
        """
        conv = conversation or self._current
        if conv is None:
            raise ValueError("No conversation to export")

        out_path = filepath or str(self.history_dir / f"{conv.id}_export.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, indent=2, ensure_ascii=False)

        return out_path

    # ------------------------------------------------------------------
    # Auto-trim
    # ------------------------------------------------------------------

    def auto_trim(self) -> int:
        """Trim old conversation files to stay within *max_history*.

        Returns:
            The number of files removed.
        """
        files = sorted(self.history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if len(files) <= self.max_history:
            return 0

        to_remove = files[: len(files) - self.max_history]
        removed = 0
        for f in to_remove:
            try:
                f.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_message_dicts(self) -> List[Dict[str, str]]:
        """Return current conversation messages as simple {role, content} dicts.

        Useful for feeding into the InferenceEngine.chat() method.
        """
        if self._current is None:
            return []
        return [{"role": m.role, "content": m.content} for m in self._current.messages]
