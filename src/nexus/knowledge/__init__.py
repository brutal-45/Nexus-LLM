"""
Nexus LLM — Knowledge Module
==============================

Production-grade knowledge management for retrieval-augmented generation,
including dense/sparse/hybrid retrieval, vector stores, knowledge graphs,
reranking, document processing, chunking, and end-to-end RAG pipelines.

Public API
----------
Config:
    RetrievalConfig, EmbeddingConfig, VectorStoreConfig, RAGConfig,
    KnowledgeGraphConfig, ChunkingConfig

Retrieval:
    DenseRetriever, SparseRetriever, HybridRetriever, ColBERTRetriever,
    MultiVectorRetriever, CrossEncoderReranker, RetrievalPipeline

Reranking:
    BaseReranker, CrossEncoderReranker, T5Reranker, ColBERTLateInteraction,
    MMRReranker, KnowledgeDistilledReranker, ListWiseReranker, RerankerEnsemble

Vector Store:
    BaseVectorStore, SimpleVectorStore, HNSWIndex, IVFIndex,
    ProductQuantizationIndex, DiskANNIndex, VectorStoreManager

Embeddings:
    BaseEmbeddingModel, SimpleEmbedding, TransformerEmbedding,
    MatryoshkaEmbedding, ContrieverEmbedding, E5Embedding, BGEEmbedding,
    EmbeddingTrainer

Knowledge Graph:
    Entity, Relation, KnowledgeGraph, GraphEmbedding, KGReasoner,
    KGRetriever, GraphVisualizer

RAG Pipeline:
    RAGPipeline, ContextBuilder, DocumentIndexer, SourceTracker,
    QueryRewriter, HybridRAG, MultiStepRAG, FaithfulnessChecker

Document Processing:
    DocumentLoader, DocumentCleaner, DocumentSplitter, DocumentTransformer

Chunking:
    FixedChunker, RecursiveChunker, SemanticChunker,
    SlidingWindowChunker, DocumentStructureChunker, AgenticChunker
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    # ── Config ──────────────────────────────────────────────────────────
    "RetrievalConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "RAGConfig",
    "KnowledgeGraphConfig",
    "ChunkingConfig",
    # ── Retrieval ───────────────────────────────────────────────────────
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "ColBERTRetriever",
    "MultiVectorRetriever",
    "CrossEncoderReranker",
    "RetrievalPipeline",
    # ── Reranking ───────────────────────────────────────────────────────
    "BaseReranker",
    "CrossEncoderReranker",
    "T5Reranker",
    "ColBERTLateInteraction",
    "MMRReranker",
    "KnowledgeDistilledReranker",
    "ListWiseReranker",
    "RerankerEnsemble",
    # ── Vector Store ────────────────────────────────────────────────────
    "BaseVectorStore",
    "SimpleVectorStore",
    "HNSWIndex",
    "IVFIndex",
    "ProductQuantizationIndex",
    "DiskANNIndex",
    "VectorStoreManager",
    # ── Embeddings ──────────────────────────────────────────────────────
    "BaseEmbeddingModel",
    "SimpleEmbedding",
    "TransformerEmbedding",
    "MatryoshkaEmbedding",
    "ContrieverEmbedding",
    "E5Embedding",
    "BGEEmbedding",
    "EmbeddingTrainer",
    # ── Knowledge Graph ─────────────────────────────────────────────────
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "GraphEmbedding",
    "TransE",
    "TransH",
    "TransR",
    "ComplEx",
    "RotatE",
    "KGReasoner",
    "KGRetriever",
    "GraphVisualizer",
    # ── RAG Pipeline ────────────────────────────────────────────────────
    "RAGPipeline",
    "ContextBuilder",
    "DocumentIndexer",
    "SourceTracker",
    "QueryRewriter",
    "HybridRAG",
    "MultiStepRAG",
    "FaithfulnessChecker",
    # ── Document Processing ─────────────────────────────────────────────
    "DocumentLoader",
    "DocumentCleaner",
    "DocumentSplitter",
    "DocumentTransformer",
    # ── Chunking ────────────────────────────────────────────────────────
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    "DocumentStructureChunker",
    "AgenticChunker",
]

# ── Module-level logger ────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger("nexus.knowledge")


# ── Lazy imports for fast initialisation ───────────────────────────────────
def __getattr__(name: str) -> Any:
    """Lazy-load heavy sub-modules on first access."""
    _config_map: Dict[str, Tuple[str, str]] = {
        "RetrievalConfig": ("knowledge_config", "RetrievalConfig"),
        "EmbeddingConfig": ("knowledge_config", "EmbeddingConfig"),
        "VectorStoreConfig": ("knowledge_config", "VectorStoreConfig"),
        "RAGConfig": ("knowledge_config", "RAGConfig"),
        "KnowledgeGraphConfig": ("knowledge_config", "KnowledgeGraphConfig"),
        "ChunkingConfig": ("knowledge_config", "ChunkingConfig"),
    }
    _retrieval_map: Dict[str, Tuple[str, str]] = {
        "DenseRetriever": ("retrieval", "DenseRetriever"),
        "SparseRetriever": ("retrieval", "SparseRetriever"),
        "HybridRetriever": ("retrieval", "HybridRetriever"),
        "ColBERTRetriever": ("retrieval", "ColBERTRetriever"),
        "MultiVectorRetriever": ("retrieval", "MultiVectorRetriever"),
        "CrossEncoderReranker": ("retrieval", "CrossEncoderReranker"),
        "RetrievalPipeline": ("retrieval", "RetrievalPipeline"),
    }
    _reranking_map: Dict[str, Tuple[str, str]] = {
        "BaseReranker": ("reranking", "BaseReranker"),
        "CrossEncoderReranker": ("reranking", "CrossEncoderReranker"),
        "T5Reranker": ("reranking", "T5Reranker"),
        "ColBERTLateInteraction": ("reranking", "ColBERTLateInteraction"),
        "MMRReranker": ("reranking", "MMRReranker"),
        "KnowledgeDistilledReranker": ("reranking", "KnowledgeDistilledReranker"),
        "ListWiseReranker": ("reranking", "ListWiseReranker"),
        "RerankerEnsemble": ("reranking", "RerankerEnsemble"),
    }
    _vector_store_map: Dict[str, Tuple[str, str]] = {
        "BaseVectorStore": ("vector_store", "BaseVectorStore"),
        "SimpleVectorStore": ("vector_store", "SimpleVectorStore"),
        "HNSWIndex": ("vector_store", "HNSWIndex"),
        "IVFIndex": ("vector_store", "IVFIndex"),
        "ProductQuantizationIndex": ("vector_store", "ProductQuantizationIndex"),
        "DiskANNIndex": ("vector_store", "DiskANNIndex"),
        "VectorStoreManager": ("vector_store", "VectorStoreManager"),
    }
    _embeddings_map: Dict[str, Tuple[str, str]] = {
        "BaseEmbeddingModel": ("embeddings_advanced", "BaseEmbeddingModel"),
        "SimpleEmbedding": ("embeddings_advanced", "SimpleEmbedding"),
        "TransformerEmbedding": ("embeddings_advanced", "TransformerEmbedding"),
        "MatryoshkaEmbedding": ("embeddings_advanced", "MatryoshkaEmbedding"),
        "ContrieverEmbedding": ("embeddings_advanced", "ContrieverEmbedding"),
        "E5Embedding": ("embeddings_advanced", "E5Embedding"),
        "BGEEmbedding": ("embeddings_advanced", "BGEEmbedding"),
        "EmbeddingTrainer": ("embeddings_advanced", "EmbeddingTrainer"),
    }
    _kg_map: Dict[str, Tuple[str, str]] = {
        "Entity": ("knowledge_graph", "Entity"),
        "Relation": ("knowledge_graph", "Relation"),
        "KnowledgeGraph": ("knowledge_graph", "KnowledgeGraph"),
        "GraphEmbedding": ("knowledge_graph", "GraphEmbedding"),
        "TransE": ("knowledge_graph", "TransE"),
        "TransH": ("knowledge_graph", "TransH"),
        "TransR": ("knowledge_graph", "TransR"),
        "ComplEx": ("knowledge_graph", "ComplEx"),
        "RotatE": ("knowledge_graph", "RotatE"),
        "KGReasoner": ("knowledge_graph", "KGReasoner"),
        "KGRetriever": ("knowledge_graph", "KGRetriever"),
        "GraphVisualizer": ("knowledge_graph", "GraphVisualizer"),
    }
    _rag_map: Dict[str, Tuple[str, str]] = {
        "RAGPipeline": ("rag_pipeline", "RAGPipeline"),
        "ContextBuilder": ("rag_pipeline", "ContextBuilder"),
        "DocumentIndexer": ("rag_pipeline", "DocumentIndexer"),
        "SourceTracker": ("rag_pipeline", "SourceTracker"),
        "QueryRewriter": ("rag_pipeline", "QueryRewriter"),
        "HybridRAG": ("rag_pipeline", "HybridRAG"),
        "MultiStepRAG": ("rag_pipeline", "MultiStepRAG"),
        "FaithfulnessChecker": ("rag_pipeline", "FaithfulnessChecker"),
    }
    _doc_map: Dict[str, Tuple[str, str]] = {
        "DocumentLoader": ("document_processing", "DocumentLoader"),
        "DocumentCleaner": ("document_processing", "DocumentCleaner"),
        "DocumentSplitter": ("document_processing", "DocumentSplitter"),
        "DocumentTransformer": ("document_processing", "DocumentTransformer"),
    }
    _chunking_map: Dict[str, Tuple[str, str]] = {
        "FixedChunker": ("chunking", "FixedChunker"),
        "RecursiveChunker": ("chunking", "RecursiveChunker"),
        "SemanticChunker": ("chunking", "SemanticChunker"),
        "SlidingWindowChunker": ("chunking", "SlidingWindowChunker"),
        "DocumentStructureChunker": ("chunking", "DocumentStructureChunker"),
        "AgenticChunker": ("chunking", "AgenticChunker"),
    }

    # Combine all lookup tables
    all_maps: Dict[str, Tuple[str, str]] = {}
    all_maps.update(_config_map)
    all_maps.update(_retrieval_map)
    all_maps.update(_reranking_map)
    all_maps.update(_vector_store_map)
    all_maps.update(_embeddings_map)
    all_maps.update(_kg_map)
    all_maps.update(_rag_map)
    all_maps.update(_doc_map)
    all_maps.update(_chunking_map)

    if name in all_maps:
        module_name, attr_name = all_maps[name]
        import importlib

        mod = importlib.import_module(f".{module_name}", package=__name__)
        return getattr(mod, attr_name)

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


# ── Version ────────────────────────────────────────────────────────────────
__version__: str = "0.1.0"
