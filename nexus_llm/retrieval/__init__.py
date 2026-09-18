"""Retrieval module for Nexus-LLM.

Provides document retrieval via keyword search, semantic (vector)
search, and hybrid search with reciprocal rank fusion.
"""

from nexus_llm.retrieval.engine import RetrievalEngine
from nexus_llm.retrieval.hybrid import HybridRetriever
from nexus_llm.retrieval.vector_index import VectorIndex

__all__ = [
    "HybridRetriever",
    "RetrievalEngine",
    "VectorIndex",
]
