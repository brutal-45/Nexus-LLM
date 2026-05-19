"""Text chunking strategies for document splitting.

Provides multiple chunking approaches including fixed-size, sentence-based,
paragraph-based, and semantic chunking with configurable overlap.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    start_index: int = 0
    end_index: int = 0
    source_doc_id: Optional[str] = None

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id[:8]}..., text='{preview}...', len={len(self)})"


class TextChunker(ABC):
    """Abstract base class for text chunking strategies."""

    def __init__(self, overlap: int = 0):
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None, source_doc_id: Optional[str] = None) -> List[Chunk]:
        """Split text into chunks."""
        ...


class FixedSizeChunker(TextChunker):
    """Chunk text into fixed-size pieces with optional overlap.

    Splits text based on a specified character count, optionally
    overlapping chunks to preserve context at boundaries.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50, separator: str = " "):
        super().__init__(overlap=overlap)
        self.chunk_size = chunk_size
        self.separator = separator

    def chunk(self, text: str, metadata: Optional[dict] = None, source_doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        step = self.chunk_size - self.overlap

        if step <= 0:
            step = self.chunk_size

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at the last separator within the window
            if end < len(text):
                last_sep = text.rfind(self.separator, start + self.chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_meta = (metadata or {}).copy()
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=chunk_meta,
                        start_index=start,
                        end_index=end,
                        source_doc_id=source_doc_id,
                    )
                )

            start += step
            if start >= len(text):
                break
            # Adjust start to avoid splitting mid-word
            if self.separator and start < len(text):
                next_sep = text.find(self.separator, start)
                if next_sep != -1 and next_sep - start < self.overlap:
                    start = next_sep + 1

        return chunks


class SentenceChunker(TextChunker):
    """Chunk text by sentences with configurable maximum chunk size.

    Splits text on sentence boundaries, then groups sentences
    together until the maximum chunk size is reached.
    """

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+|(?<=\n)\s*", re.MULTILINE
    )

    def __init__(self, max_chunk_size: int = 512, overlap_sentences: int = 1):
        super().__init__(overlap=0)
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        raw_sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def chunk(self, text: str, metadata: Optional[dict] = None, source_doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_length = 0
        char_offset = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            if current_length + sentence_len > self.max_chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=(metadata or {}).copy(),
                        start_index=char_offset,
                        end_index=char_offset + len(chunk_text),
                        source_doc_id=source_doc_id,
                    )
                )
                # Apply overlap: keep last N sentences
                overlap_start = max(0, len(current_sentences) - self.overlap_sentences)
                char_offset += len(" ".join(current_sentences[:overlap_start])) + 1
                current_sentences = current_sentences[overlap_start:]
                current_length = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sentence_len + 1  # +1 for space

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=(metadata or {}).copy(),
                    start_index=char_offset,
                    end_index=char_offset + len(chunk_text),
                    source_doc_id=source_doc_id,
                )
            )

        return chunks


class ParagraphChunker(TextChunker):
    """Chunk text by paragraphs, optionally splitting large paragraphs.

    Identifies paragraph boundaries (double newlines or blank lines)
    and creates chunks. Paragraphs exceeding max_chunk_size are
    further split using a fixed-size approach.
    """

    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    def __init__(self, max_chunk_size: int = 1024, overlap: int = 0):
        super().__init__(overlap=overlap)
        self.max_chunk_size = max_chunk_size
        self._fallback_chunker = FixedSizeChunker(
            chunk_size=max_chunk_size, overlap=overlap
        )

    def chunk(self, text: str, metadata: Optional[dict] = None, source_doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        offset = 0

        for para in paragraphs:
            start_in_text = text.find(para, offset)
            if start_in_text == -1:
                start_in_text = offset

            if len(para) <= self.max_chunk_size:
                chunks.append(
                    Chunk(
                        text=para,
                        metadata=(metadata or {}).copy(),
                        start_index=start_in_text,
                        end_index=start_in_text + len(para),
                        source_doc_id=source_doc_id,
                    )
                )
            else:
                # Split oversized paragraphs with fallback chunker
                sub_chunks = self._fallback_chunker.chunk(
                    para, metadata=metadata, source_doc_id=source_doc_id
                )
                for sc in sub_chunks:
                    sc.start_index += start_in_text
                    sc.end_index += start_in_text
                chunks.extend(sub_chunks)

            offset = start_in_text + len(para)

        return chunks


class SemanticChunker(TextChunker):
    """Chunk text based on semantic similarity between sentences.

    Groups consecutive sentences that are semantically similar.
    Falls back to sentence-based chunking when embedding function
    is not available. Uses a simple heuristic based on shared
    vocabulary when no embedding model is provided.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        similarity_threshold: float = 0.3,
        embedding_fn: Optional[callable] = None,
    ):
        super().__init__(overlap=0)
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn

    def _token_overlap_similarity(self, sent_a: str, sent_b: str) -> float:
        """Compute a simple token-overlap Jaccard similarity between two sentences."""
        tokens_a = set(sent_a.lower().split())
        tokens_b = set(sent_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _compute_similarity(self, sent_a: str, sent_b: str) -> float:
        """Compute similarity between two sentences using embeddings or token overlap."""
        if self.embedding_fn is not None:
            try:
                import numpy as np

                emb_a = self.embedding_fn([sent_a])
                emb_b = self.embedding_fn([sent_b])
                if isinstance(emb_a, list):
                    emb_a = np.array(emb_a)
                    emb_b = np.array(emb_b)
                norm_a = np.linalg.norm(emb_a, axis=-1, keepdims=True)
                norm_b = np.linalg.norm(emb_b, axis=-1, keepdims=True)
                norm_a = np.where(norm_a == 0, 1, norm_a)
                norm_b = np.where(norm_b == 0, 1, norm_b)
                cos_sim = np.sum(emb_a / norm_a * emb_b / norm_b, axis=-1)
                return float(cos_sim)
            except Exception:
                pass
        return self._token_overlap_similarity(sent_a, sent_b)

    def chunk(self, text: str, metadata: Optional[dict] = None, source_doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        sentence_chunker = SentenceChunker(max_chunk_size=self.max_chunk_size)
        sentences = sentence_chunker._split_sentences(text)

        if len(sentences) <= 1:
            return [
                Chunk(
                    text=text.strip(),
                    metadata=(metadata or {}).copy(),
                    start_index=0,
                    end_index=len(text),
                    source_doc_id=source_doc_id,
                )
            ]

        # Group sentences by semantic similarity
        groups: List[List[str]] = [[sentences[0]]]
        group_lengths: List[int] = [len(sentences[0])]

        for i in range(1, len(sentences)):
            sim = self._compute_similarity(sentences[i - 1], sentences[i])
            current_len = group_lengths[-1] + len(sentences[i]) + 1

            if sim >= self.similarity_threshold and current_len <= self.max_chunk_size:
                groups[-1].append(sentences[i])
                group_lengths[-1] = current_len
            else:
                groups.append([sentences[i]])
                group_lengths.append(len(sentences[i]))

        # Convert groups to chunks
        chunks: List[Chunk] = []
        offset = 0

        for group in groups:
            chunk_text = " ".join(group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=(metadata or {}).copy(),
                    start_index=offset,
                    end_index=offset + len(chunk_text),
                    source_doc_id=source_doc_id,
                )
            )
            offset += len(chunk_text) + 1

        return chunks
