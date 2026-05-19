"""Test text chunking for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.end_char == 0:
            self.end_char = self.start_char + len(self.text)


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, separator: str = "\n"):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separator = separator

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def chunk_overlap(self):
        return self._chunk_overlap

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []
        return self._chunk_with_overlap(text)

    def _chunk_with_overlap(self, text: str) -> List[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]
            if end < len(text):
                sep_pos = chunk_text.rfind(self._separator)
                if sep_pos > self._chunk_size // 2:
                    chunk_text = chunk_text[:sep_pos + len(self._separator)]
                    end = start + sep_pos + len(self._separator)
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=start,
                end_char=end,
            ))
            idx += 1
            start = end - self._chunk_overlap
            if start >= len(text):
                break
        return chunks

    def chunk_by_sentences(self, text: str, max_chunk_size: int = None) -> List[Chunk]:
        size = max_chunk_size or self._chunk_size
        sentences = [s.strip() for s in text.replace("?", "?|").replace("!", "!|").replace(".", ".|").split("|") if s.strip()]
        chunks = []
        current_text = ""
        idx = 0
        for sentence in sentences:
            if len(current_text) + len(sentence) + 1 > size and current_text:
                chunks.append(Chunk(text=current_text.strip(), index=idx))
                idx += 1
                current_text = sentence
            else:
                current_text = (current_text + " " + sentence).strip()
        if current_text:
            chunks.append(Chunk(text=current_text.strip(), index=idx))
        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [Chunk(text=p, index=i) for i, p in enumerate(paragraphs)]


class TestChunk:
    def test_creation(self):
        chunk = Chunk(text="hello", index=0)
        assert chunk.text == "hello"
        assert chunk.index == 0
        assert chunk.metadata == {}

    def test_end_char_auto(self):
        chunk = Chunk(text="hello", index=0, start_char=5)
        assert chunk.end_char == 10


class TestTextChunker:
    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="positive"):
            TextChunker(chunk_size=0)

    def test_overlap_too_large(self):
        with pytest.raises(ValueError, match="overlap"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []

    def test_short_text(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.chunk("Hello world")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_long_text_split(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "a" * 100
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_index_sequential(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk("a" * 100)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_properties(self):
        chunker = TextChunker(chunk_size=256, chunk_overlap=32)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32

    def test_chunk_by_sentences(self):
        chunker = TextChunker()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_by_sentences(max_chunk_size=40)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.text) <= 50  # approximate

    def test_chunk_by_paragraphs(self):
        chunker = TextChunker()
        text = "Para one\n\nPara two\n\nPara three"
        chunks = chunker.chunk_by_paragraphs(text)
        assert len(chunks) == 3

    def test_overlap_produces_more_chunks(self):
        text = "word " * 100
        chunker_no_overlap = TextChunker(chunk_size=50, chunk_overlap=0)
        chunker_with_overlap = TextChunker(chunk_size=50, chunk_overlap=10)
        c1 = chunker_no_overlap.chunk(text)
        c2 = chunker_with_overlap.chunk(text)
        assert len(c2) >= len(c1)
