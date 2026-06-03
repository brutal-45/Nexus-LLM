"""RAG (Retrieval-Augmented Generation) module for Nexus-LLM.

Provides document storage, chunking, indexing, retrieval, and a
complete RAG pipeline.
"""

from nexus_llm.rag.engine import RAGEngine
from nexus_llm.rag.document_store import DocumentStore
from nexus_llm.rag.retriever import Retriever
from nexus_llm.rag.indexer import Indexer
from nexus_llm.rag.chunker import Chunker
from nexus_llm.rag.pipeline import RAGPipeline

__all__ = [
    "RAGEngine",
    "DocumentStore",
    "Retriever",
    "Indexer",
    "Chunker",
    "RAGPipeline",
]
