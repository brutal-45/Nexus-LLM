"""Chat History Manager - Stores, retrieves, and manages conversation history."""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    generation_time: float = 0.0


class ChatHistory:
    """
    Manages chat history with persistence, search, and export capabilities.
    Supports multiple conversation sessions.
    """

    def __init__(self, history_dir: str = "./chat_history", max_turns: int = 20):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns

        self._current_session: List[Message] = []
        self._session_id: str = self._generate_session_id()
        self._sessions_index: Dict[str, Dict[str, Any]] = self._load_sessions_index()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return time.strftime("%Y%m%d_%H%M%S")

    def _load_sessions_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the sessions index file."""
        index_path = self.history_dir / "sessions_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_sessions_index(self) -> None:
        """Save the sessions index file."""
        index_path = self.history_dir / "sessions_index.json"
        with open(index_path, "w") as f:
            json.dump(self._sessions_index, f, indent=2, default=str)

    def add_message(
        self,
        role: str,
        content: str,
        token_count: int = 0,
        generation_time: float = 0.0,
    ) -> Message:
        """Add a message to the current session."""
        msg = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            token_count=token_count,
            generation_time=generation_time,
        )
        self._current_session.append(msg)

        # Trim history if exceeding max_turns
        while len(self._current_session) > self.max_turns * 2:
            # Remove oldest user-assistant pair
            if len(self._current_session) >= 2:
                self._current_session.pop(0)
                self._current_session.pop(0)
            else:
                break

        return msg

    def get_messages(self, include_system: bool = False) -> List[Dict[str, str]]:
        """Get messages in the format expected by the inference engine."""
        messages = []
        for msg in self._current_session:
            if not include_system and msg.role == "system":
                continue
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def get_last_response(self) -> Optional[str]:
        """Get the last assistant response."""
        for msg in reversed(self._current_session):
            if msg.role == "assistant":
                return msg.content
        return None

    def clear(self) -> None:
        """Clear the current session."""
        self.save_session()
        self._current_session = []
        self._session_id = self._generate_session_id()

    def save_session(self) -> None:
        """Save the current session to disk."""
        if not self._current_session:
            return

        session_file = self.history_dir / f"session_{self._session_id}.json"
        session_data = {
            "session_id": self._session_id,
            "created_at": self._current_session[0].timestamp if self._current_session else None,
            "updated_at": time.time(),
            "message_count": len(self._current_session),
            "messages": [asdict(msg) for msg in self._current_session],
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        # Update sessions index
        self._sessions_index[self._session_id] = {
            "message_count": len(self._current_session),
            "created_at": session_data["created_at"],
            "updated_at": session_data["updated_at"],
            "file": str(session_file),
        }
        self._save_sessions_index()

        logger.info(f"Session saved: {self._session_id}")

    def load_session(self, session_id: str) -> bool:
        """Load a previous session by ID."""
        session_file = self.history_dir / f"session_{session_id}.json"
        if not session_file.exists():
            logger.warning(f"Session not found: {session_id}")
            return False

        try:
            with open(session_file, "r") as f:
                data = json.load(f)

            self._current_session = []
            for msg_data in data.get("messages", []):
                self._current_session.append(Message(**msg_data))
            self._session_id = session_id

            logger.info(f"Session loaded: {session_id} ({len(self._current_session)} messages)")
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions."""
        sessions = []
        for sid, info in sorted(
            self._sessions_index.items(), key=lambda x: x[1].get("updated_at", 0), reverse=True
        ):
            sessions.append({"session_id": sid, **info})
        return sessions

    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search through all conversation history."""
        results = []
        query_lower = query.lower()

        for sid, info in self._sessions_index.items():
            session_file = self.history_dir / f"session_{sid}.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file, "r") as f:
                    data = json.load(f)

                for msg_data in data.get("messages", []):
                    if query_lower in msg_data.get("content", "").lower():
                        results.append({
                            "session_id": sid,
                            "role": msg_data["role"],
                            "content": msg_data["content"][:200],
                            "timestamp": msg_data["timestamp"],
                        })
            except (json.JSONDecodeError, IOError):
                continue

        return results

    def export_session(self, session_id: Optional[str] = None, format: str = "json") -> str:
        """Export a session to a specific format."""
        if session_id:
            session_file = self.history_dir / f"session_{session_id}.json"
        else:
            session_file = self.history_dir / f"session_{self._session_id}.json"

        if not session_file.exists():
            return ""

        with open(session_file, "r") as f:
            data = json.load(f)

        if format == "markdown" or format == "md":
            lines = [f"# Chat Session: {data['session_id']}\n"]
            for msg in data.get("messages", []):
                role = msg["role"].capitalize()
                lines.append(f"**{role}:** {msg['content']}\n")
            return "\n".join(lines)
        else:
            return json.dumps(data, indent=2)

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._session_id

    @property
    def message_count(self) -> int:
        """Get the number of messages in the current session."""
        return len(self._current_session)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used in the current session."""
        return sum(msg.token_count for msg in self._current_session)
