"""Chunker for Nexus-LLM RAG.

Splits text into chunks using multiple strategies: fixed-size, sentence,
paragraph, and semantic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chunk data structure
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk of text produced by the chunker.

    Attributes:
        content: The chunk text.
        index: Position of this chunk in the original sequence.
        start_char: Starting character offset in the original text.
        end_char: Ending character offset (exclusive) in the original text.
        metadata: Optional key-value metadata.
    """

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Optional[dict] = None

    def __str__(self) -> str:
        return self.content


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using punctuation heuristics."""
    parts = _SENTENCE_ENDINGS.split(text.strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Split text into overlapping chunks using various strategies.

    Args:
        strategy: The chunking strategy.
        chunk_size: Target size for each chunk (in characters).
        overlap: Number of overlapping characters between chunks.
    """

    def __init__(
        self,
        strategy: str = "fixed_size",
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> None:
        self.strategy = ChunkStrategy(strategy)
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(
            "Chunker initialised (strategy=%s, size=%d, overlap=%d)",
            self.strategy.value, chunk_size, overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, strategy: Optional[str] = None) -> List[Chunk]:
        """Split *text* into chunks.

        Args:
            text: The text to chunk.
            strategy: Override the default strategy for this call.

        Returns:
            A list of :class:`Chunk` objects in order.
        """
        strat = ChunkStrategy(strategy) if strategy else self.strategy

        if strat == ChunkStrategy.FIXED_SIZE:
            chunks = self._fixed_size(text)
        elif strat == ChunkStrategy.SENTENCE:
            chunks = self._sentence(text)
        elif strat == ChunkStrategy.PARAGRAPH:
            chunks = self._paragraph(text)
        elif strat == ChunkStrategy.SEMANTIC:
            chunks = self._semantic(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strat}")

        logger.debug(
            "Chunked text into %d chunk(s) using strategy=%s",
            len(chunks), strat.value,
        )
        return chunks

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fixed_size(self, text: str) -> List[Chunk]:
        """Fixed-size chunks with optional overlap."""
        chunks: List[Chunk] = []
        step = max(self.chunk_size - self.overlap, 1)
        idx = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "fixed_size"},
                ))
                idx += 1
            start += step

        return chunks

    def _sentence(self, text: str) -> List[Chunk]:
        """Sentence-level chunks, grouping sentences up to chunk_size."""
        sentences = _split_sentences(text)
        return self._group_units(sentences, "sentence")

    def _paragraph(self, text: str) -> List[Chunk]:
        """Paragraph-level chunks (split on double newlines)."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return self._group_units(paragraphs, "paragraph")

    def _semantic(self, text: str) -> List[Chunk]:
        """Semantic chunking approximation using topic-shift heuristics.

        Splits on paragraph boundaries and further on sentences that
        start with discourse markers (e.g. "However", "In contrast").
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: List[str] = []

        topic_markers = re.compile(
            r"^(?:however|in\s+contrast|on\s+the\s+other\s+hand|meanwhile|furthermore|moreover|in\s+addition|as\s+a\s+result|consequently|therefore|thus|nevertheless|nonetheless|importantly|notably|specifically|for\s+example|for\s+instance)",
            re.IGNORECASE,
        )

        for para in paragraphs:
            sentences = _split_sentences(para)
            current: List[str] = []
            for sent in sentences:
                if current and topic_markers.match(sent.strip()):
                    units.append(" ".join(current))
                    current = []
                current.append(sent)
            if current:
                units.append(" ".join(current))

        return self._group_units(units, "semantic")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _group_units(self, units: List[str], strategy_name: str) -> List[Chunk]:
        """Group small text units into chunks of up to ``chunk_size`` chars."""
        chunks: List[Chunk] = []
        current_parts: List[str] = []
        current_len = 0
        char_offset = 0
        idx = 0

        for unit in units:
            unit_len = len(unit)
            if current_len + unit_len + 1 > self.chunk_size and current_parts:
                chunk_text = " ".join(current_parts)
                chunks.append(Chunk(
                    content=chunk_text,
                    index=idx,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    metadata={"strategy": strategy_name},
                ))
                char_offset += len(chunk_text) + 1
                idx += 1
                # Handle overlap: keep the last unit if overlap > 0
                if self.overlap > 0 and current_parts:
                    overlap_text = current_parts[-1]
                    current_parts = [overlap_text, unit] if overlap_text else [unit]
                    current_len = len(overlap_text) + unit_len + 1
                else:
                    current_parts = [unit]
                    current_len = unit_len
            else:
                current_parts.append(unit)
                current_len += unit_len + 1

        if current_parts:
            chunk_text = " ".join(current_parts)
            chunks.append(Chunk(
                content=chunk_text,
                index=idx,
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                metadata={"strategy": strategy_name},
            ))

        return chunks
