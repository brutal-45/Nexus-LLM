"""Retrieval module for Nexus-LLM.

Provides document retrieval via keyword search, semantic (vector)
search, and hybrid search with reciprocal rank fusion.
"""

from nexus_llm.retrieval.engine import RetrievalEngine
from nexus_llm.retrieval.vector_index import VectorIndex
from nexus_llm.retrieval.hybrid import HybridRetriever

__all__ = [
    "RetrievalEngine",
    "VectorIndex",
    "HybridRetriever",
]
