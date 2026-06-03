"""SummaryMemory — auto-summarizing hierarchical memory."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SummaryNode:
    """A single summary entry."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    level: int = 0  # 0 = raw, 1 = first-level summary, 2 = meta-summary, …
    source_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "level": self.level,
            "source_ids": self.source_ids,
            "created_at": self.created_at.isoformat(),
        }


class SummaryMemory:
    """Memory that auto-summarizes content when it exceeds a threshold.

    Content is stored as level-0 (raw) entries.  When the total number of
    level-0 entries exceeds *summarize_threshold*, a summarizer function is
    invoked to condense them into a level-1 summary.  This process repeats
    hierarchically: when enough level-*N* summaries accumulate they are
    condensed into a level-*N+1* summary.

    Parameters
    ----------
    summarize_threshold:
        Number of entries at the same level that triggers auto-summarization.
    summarizer:
        A callable ``Callable[[List[str]], str]`` that receives a list of
        content strings and returns a single summary string.  Defaults to a
        simple concatenation-based summarizer.
    id:
        Optional explicit identifier.
    """

    def __init__(
        self,
        summarize_threshold: int = 10,
        summarizer: Optional[Any] = None,
        id: Optional[str] = None,
    ) -> None:
        if summarize_threshold < 2:
            raise ValueError("summarize_threshold must be >= 2")
        self.id: str = id or uuid.uuid4().hex[:12]
        self.summarize_threshold = summarize_threshold
        self._summarizer = summarizer or self._default_summarizer
        self._entries: List[SummaryNode] = []

    # ------------------------------------------------------------------
    # Content management
    # ------------------------------------------------------------------

    def add(self, content: str) -> SummaryNode:
        """Add raw content and trigger auto-summarization if needed.

        Returns
        -------
        The newly created :class:`SummaryNode`.
        """
        node = SummaryNode(content=content, level=0)
        self._entries.append(node)
        logger.debug("SummaryMemory %s: added raw entry %s", self.id, node.id)
        self._auto_summarize()
        return node

    def get_summary(self, level: int = 1) -> str:
        """Return the concatenated summaries at the given *level*.

        If no summaries exist at *level*, the raw (level-0) entries are
        returned instead.
        """
        nodes = [n for n in self._entries if n.level == level]
        if not nodes:
            nodes = [n for n in self._entries if n.level == 0]
        return "\n---\n".join(n.content for n in nodes)

    def get_all_entries(self) -> List[SummaryNode]:
        """Return a copy of all entries (raw + summaries)."""
        return list(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Hierarchical summarization
    # ------------------------------------------------------------------

    def _auto_summarize(self) -> None:
        """Recursively summarize levels that exceed the threshold."""
        for level in range(100):  # guard against runaway recursion
            level_nodes = [n for n in self._entries if n.level == level]
            if len(level_nodes) < self.summarize_threshold:
                break
            # Create a summary at level+1 from all level entries.
            contents = [n.content for n in level_nodes]
            summary_text = self._summarizer(contents)
            new_node = SummaryNode(
                content=summary_text,
                level=level + 1,
                source_ids=[n.id for n in level_nodes],
            )
            # Remove the summarized level entries.
            self._entries = [n for n in self._entries if n.level != level]
            self._entries.append(new_node)
            logger.info(
                "SummaryMemory %s: summarized %d level-%d entries → %s (level %d)",
                self.id, len(level_nodes), level, new_node.id, level + 1,
            )

    # ------------------------------------------------------------------
    # Default summarizer
    # ------------------------------------------------------------------

    @staticmethod
    def _default_summarizer(contents: List[str]) -> str:
        """Simple concatenation-based summarizer with a header."""
        header = f"[Auto-summary of {len(contents)} entries]"
        # Truncate each entry to ~200 chars for a compact summary.
        truncated = [c[:200] + ("…" if len(c) > 200 else "") for c in contents]
        return f"{header}\n" + "\n".join(truncated)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        levels: Dict[int, int] = {}
        for n in self._entries:
            levels[n.level] = levels.get(n.level, 0) + 1
        return f"SummaryMemory(id={self.id!r}, levels={levels})"
