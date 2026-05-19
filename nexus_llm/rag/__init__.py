"""Nexus-LLM RAG (Retrieval-Augmented Generation) Module.

Provides document indexing, chunking, embedding, vector storage,
retrieval, reranking, and end-to-end RAG pipeline capabilities.
"""

from nexus_llm.rag.chunker import (
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
    SemanticChunker,
    TextChunker,
)
from nexus_llm.rag.embeddings import (
    EmbeddingModel,
    HuggingFaceEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from nexus_llm.rag.indexer import DocumentIndexer, IncrementalIndexer
from nexus_llm.rag.pipeline import RAGPipeline, RAGConfig
from nexus_llm.rag.reranker import (
    CrossEncoderReranker,
    DiversityReranker,
    RelevanceReranker,
)
from nexus_llm.rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    Retriever,
    SimilarityRetriever,
)
from nexus_llm.rag.vector_store import (
    FAISSVectorStore,
    VectorStore,
)

__all__ = [
    # Chunker
    "FixedSizeChunker",
    "ParagraphChunker",
    "SentenceChunker",
    "SemanticChunker",
    "TextChunker",
    # Embeddings
    "EmbeddingModel",
    "HuggingFaceEmbeddingModel",
    "SentenceTransformerEmbeddingModel",
    # Indexer
    "DocumentIndexer",
    "IncrementalIndexer",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
    # Reranker
    "CrossEncoderReranker",
    "DiversityReranker",
    "RelevanceReranker",
    # Retriever
    "BM25Retriever",
    "HybridRetriever",
    "Retriever",
    "SimilarityRetriever",
    # Vector Store
    "FAISSVectorStore",
    "VectorStore",
]
