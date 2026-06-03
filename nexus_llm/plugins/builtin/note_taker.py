"""Note taker plugin for saving, organizing, and searching notes.

A builtin plugin providing a note-taking system with categories,
tags, full-text search, and export capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


@dataclass
class Note:
    """A single note with metadata."""

    note_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    title: str = ""
    content: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    pinned: bool = False

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        return cls(
            note_id=data.get("note_id", str(uuid.uuid4())[:12]),
            title=data.get("title", ""),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            pinned=data.get("pinned", False),
        )

    def __repr__(self) -> str:
        return f"Note(id={self.note_id}, title='{self.title[:30]}...', category={self.category})"


class NoteTakerPlugin:
    """Plugin providing a note-taking system.

    Supports creating, reading, updating, and deleting notes
    with categories, tags, full-text search, and persistence.
    """

    name = "note_taker"
    version = "1.0.0"
    description = "Save, organize, and search notes"
    dependencies: List[str] = []
    tags = ["notes", "organization", "builtin"]

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        persist_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the note taker plugin.

        Args:
            hook_manager: Optional hook manager.
            persist_path: Path for persisting notes to disk.
        """
        self.hook_manager = hook_manager
        self.persist_path = Path(persist_path) if persist_path else None
        self._active = False
        self._notes: Dict[str, Note] = {}

        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def activate(self) -> None:
        """Activate the note taker plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="note_taker_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("Note taker plugin activated. %d notes loaded.", len(self._notes))

    def deactivate(self) -> None:
        """Deactivate the note taker plugin."""
        if self.persist_path:
            self._save_to_disk()
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Note taker plugin deactivated.")

    def create_note(
        self,
        title: str,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        pinned: bool = False,
    ) -> Dict[str, Any]:
        """Create a new note.

        Args:
            title: Note title.
            content: Note content.
            category: Note category.
            tags: List of tags.
            pinned: Whether the note is pinned.

        Returns:
            Dict with created note info.
        """
        note = Note(
            title=title,
            content=content,
            category=category,
            tags=tags or [],
            pinned=pinned,
        )
        self._notes[note.note_id] = note

        if self.persist_path:
            self._save_to_disk()

        logger.info("Created note '%s' (id=%s, category=%s).", title, note.note_id, category)
        return {"success": True, "note_id": note.note_id, "note": note.to_dict()}

    def read_note(self, note_id: str) -> Dict[str, Any]:
        """Read a note by ID.

        Args:
            note_id: The note ID.

        Returns:
            Dict with note data.
        """
        note = self._notes.get(note_id)
        if not note:
            return {"success": False, "error": f"Note '{note_id}' not found."}
        return {"success": True, "note": note.to_dict()}

    def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pinned: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update an existing note.

        Args:
            note_id: The note ID to update.
            title: New title (None to keep current).
            content: New content (None to keep current).
            category: New category (None to keep current).
            tags: New tags (None to keep current).
            pinned: New pinned status (None to keep current).

        Returns:
            Dict with update result.
        """
        note = self._notes.get(note_id)
        if not note:
            return {"success": False, "error": f"Note '{note_id}' not found."}

        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
        if category is not None:
            note.category = category
        if tags is not None:
            note.tags = tags
        if pinned is not None:
            note.pinned = pinned

        note.updated_at = time.time()

        if self.persist_path:
            self._save_to_disk()

        return {"success": True, "note": note.to_dict()}

    def delete_note(self, note_id: str) -> Dict[str, Any]:
        """Delete a note.

        Args:
            note_id: The note ID to delete.

        Returns:
            Dict with deletion result.
        """
        if note_id not in self._notes:
            return {"success": False, "error": f"Note '{note_id}' not found."}

        del self._notes[note_id]

        if self.persist_path:
            self._save_to_disk()

        return {"success": True, "deleted": note_id}

    def search_notes(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Search notes by content, category, and tags.

        Args:
            query: Search query string.
            category: Optional category filter.
            tags: Optional tag filter.
            top_k: Maximum results to return.

        Returns:
            Dict with search results.
        """
        query_lower = query.lower()
        scored: List[Tuple[Note, float]] = []

        for note in self._notes.values():
            # Category filter
            if category and note.category != category:
                continue

            # Tag filter
            if tags and not all(t in note.tags for t in tags):
                continue

            # Score based on relevance
            score = 0.0

            # Title match (higher weight)
            if query_lower in note.title.lower():
                score += 2.0

            # Content match
            if query_lower in note.content.lower():
                score += 1.0
                # Bonus for multiple occurrences
                occurrences = note.content.lower().count(query_lower)
                score += min(0.5, occurrences * 0.1)

            # Tag match
            for tag in note.tags:
                if query_lower in tag.lower():
                    score += 0.5

            # Pinned notes get a small boost
            if note.pinned:
                score += 0.1

            if score > 0:
                scored.append((note, score))

        # Sort by score descending, then by updated_at
        scored.sort(key=lambda x: (x[1], x[0].updated_at), reverse=True)

        results = [{"note": note.to_dict(), "score": score} for note, score in scored[:top_k]]

        return {
            "success": True,
            "results": results,
            "total_matches": len(scored),
            "returned": len(results),
        }

    def list_notes(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        pinned_only: bool = False,
        sort_by: str = "updated",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List notes with optional filtering and sorting.

        Args:
            category: Optional category filter.
            tag: Optional tag filter.
            pinned_only: Only return pinned notes.
            sort_by: Sort field ('updated', 'created', 'title').
            limit: Maximum notes to return.

        Returns:
            Dict with note list.
        """
        notes = list(self._notes.values())

        if category:
            notes = [n for n in notes if n.category == category]
        if tag:
            notes = [n for n in notes if tag in n.tags]
        if pinned_only:
            notes = [n for n in notes if n.pinned]

        # Sort
        if sort_by == "updated":
            notes.sort(key=lambda n: n.updated_at, reverse=True)
        elif sort_by == "created":
            notes.sort(key=lambda n: n.created_at, reverse=True)
        elif sort_by == "title":
            notes.sort(key=lambda n: n.title.lower())

        # Pinned notes float to top
        notes.sort(key=lambda n: n.pinned, reverse=True)

        return {
            "success": True,
            "notes": [n.to_dict() for n in notes[:limit]],
            "total": len(notes),
        }

    def get_categories(self) -> Dict[str, Any]:
        """Get all note categories with counts."""
        cat_counts: Dict[str, int] = {}
        for note in self._notes.values():
            cat_counts[note.category] = cat_counts.get(note.category, 0) + 1
        return {"success": True, "categories": cat_counts}

    def get_all_tags(self) -> Dict[str, Any]:
        """Get all tags with usage counts."""
        tag_counts: Dict[str, int] = {}
        for note in self._notes.values():
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return {"success": True, "tags": tag_counts}

    def export_notes(self, format: str = "json") -> Dict[str, Any]:
        """Export all notes.

        Args:
            format: Export format ('json' or 'text').

        Returns:
            Dict with exported data.
        """
        if format == "json":
            data = [note.to_dict() for note in self._notes.values()]
            return {"success": True, "format": "json", "data": data, "count": len(data)}
        elif format == "text":
            lines = []
            for note in self._notes.values():
                lines.append(f"=== {note.title} ===")
                lines.append(f"Category: {note.category}")
                lines.append(f"Tags: {', '.join(note.tags)}")
                lines.append(f"Content:\n{note.content}")
                lines.append("")
            return {"success": True, "format": "text", "data": "\n".join(lines)}
        else:
            return {"success": False, "error": f"Unsupported format: {format}"}

    def _save_to_disk(self) -> None:
        """Persist notes to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {nid: note.to_dict() for nid, note in self._notes.items()}
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        """Load notes from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._notes = {nid: Note.from_dict(nd) for nid, nd in data.items()}
            logger.info("Loaded %d notes from disk.", len(self._notes))
        except Exception as e:
            logger.error("Failed to load notes: %s", e)

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for note operations."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "note_create":
            title = kwargs.get("title", "Untitled")
            content = kwargs.get("content", "")
            category = kwargs.get("category", "general")
            create_result = self.create_note(title, content, category)
            return f"Note created: {create_result.get('note_id', 'unknown')}"
        elif tool_name == "note_search":
            query = kwargs.get("query", "")
            search_result = self.search_notes(query)
            if search_result["success"]:
                notes = search_result["results"]
                return "\n".join(f"- {n['note']['title']}: {n['note']['content'][:50]}..." for n in notes)
        return result
