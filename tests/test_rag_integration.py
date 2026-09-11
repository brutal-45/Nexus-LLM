"""Integration tests for the RAG package's public surface and chunking."""

from __future__ import annotations

import pytest

import nexus_llm.rag as rag
from nexus_llm.rag import Chunker, DocumentStore, Indexer, RAGEngine, RAGPipeline, Retriever


class TestPublicSurface:
    @pytest.mark.parametrize("name", ["RAGEngine", "DocumentStore", "Retriever", "Indexer", "Chunker", "RAGPipeline"])
    def test_package_exports(self, name):
        assert getattr(rag, name, None) is not None

    def test_all_names_resolve(self):
        for name in rag.__all__:
            assert hasattr(rag, name), f"__all__ lists missing name {name}"

    def test_chunk_and_document_types_are_reachable(self):
        from nexus_llm.rag.chunker import Chunk, ChunkStrategy
        from nexus_llm.rag.document_store import Document

        assert {s.value for s in ChunkStrategy} >= {"fixed_size", "sentence", "paragraph"}
        assert Chunk(content="x", index=0, start_char=0, end_char=1) is not None
        assert Document is not None


class TestChunker:
    TEXT = (
        "Retrieval augmented generation grounds a model in external documents. "
        "The indexer stores chunks. The retriever scores them. "
        "Finally the generator writes an answer with citations."
    )

    def test_fixed_size_chunks_respect_the_size_cap(self):
        chunks = Chunker(strategy="fixed_size", chunk_size=80, overlap=0).chunk(self.TEXT)
        assert len(chunks) >= 2
        assert all(len(c.content) <= 80 for c in chunks)

    def test_chunks_are_numbered_and_span_the_source(self):
        chunks = Chunker(strategy="fixed_size", chunk_size=100, overlap=0).chunk(self.TEXT)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char <= len(self.TEXT)

    def test_str(self):
        chunk = Chunker(strategy="fixed_size", chunk_size=40).chunk("abcdefghij" * 3)[0]
        assert str(chunk) == chunk.content

    def test_sentence_strategy_produces_multiple_chunks(self):
        chunks = Chunker(strategy="sentence").chunk(self.TEXT)
        assert len(chunks) > 1

    def test_paragraph_strategy_groups_blank_line_blocks(self):
        text = "first paragraph line\n\nsecond paragraph line\n\nthird paragraph line"
        chunks = Chunker(strategy="paragraph").chunk(text)
        assert len(chunks) >= 2

    def test_strategy_can_be_overridden_per_call(self):
        chunker = Chunker(strategy="fixed_size", chunk_size=50)
        assert chunker.chunk(self.TEXT, strategy="paragraph")

    def test_all_strategies_return_dataclasses(self):
        import dataclasses

        from nexus_llm.rag.chunker import Chunk

        for strategy in ("fixed_size", "sentence", "paragraph", "semantic"):
            for chunk in Chunker(strategy=strategy).chunk(self.TEXT):
                assert dataclasses.is_dataclass(chunk) and isinstance(chunk, Chunk)


class TestStoreSurface:
    """Constructors must work with no arguments for the default wiring."""

    @pytest.mark.parametrize("factory", [DocumentStore, Indexer, Retriever, RAGEngine, RAGPipeline])
    def test_constructible(self, factory):
        assert factory() is not None
