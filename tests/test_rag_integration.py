"""Tests for RAG module integration."""
import pytest

from nexus_llm.rag import (
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
    TextChunker,
    RAGPipeline,
    RAGConfig,
    VectorStore,
    DocumentIndexer,
)


class TestRAGModuleImports:
    """Test that all RAG module components can be imported."""

    def test_chunker_imports(self):
        assert FixedSizeChunker is not None
        assert ParagraphChunker is not None
        assert SentenceChunker is not None
        assert TextChunker is not None

    def test_pipeline_imports(self):
        assert RAGPipeline is not None
        assert RAGConfig is not None

    def test_vector_store_import(self):
        assert VectorStore is not None

    def test_indexer_import(self):
        assert DocumentIndexer is not None


class TestRAGChunkerIntegration:
    """Test chunker components."""

    def test_fixed_size_chunker(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        assert chunker is not None
        result = chunker.chunk("Hello world. This is a test. " * 50)
        assert isinstance(result, list)

    def test_sentence_chunker(self):
        chunker = SentenceChunker()
        assert chunker is not None

    def test_paragraph_chunker(self):
        chunker = ParagraphChunker()
        assert chunker is not None

    def test_text_chunker_is_abstract(self):
        """TextChunker is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TextChunker()


class TestRAGPipelineIntegration:
    """Test RAG pipeline creation."""

    def test_create_rag_config(self):
        config = RAGConfig()
        assert config is not None


class TestDocumentIndexerIntegration:
    """Test document indexer."""

    def test_indexer_requires_args(self):
        """DocumentIndexer requires chunker, embedding_model, and vector_store."""
        with pytest.raises(TypeError):
            DocumentIndexer()
