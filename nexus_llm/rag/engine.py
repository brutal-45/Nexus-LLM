"""RAG engine for Nexus-LLM.

High-level interface that wraps the full RAG pipeline with a simple
query / add_documents API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nexus_llm.rag.document_store import Document, DocumentStore
from nexus_llm.rag.retriever import Retriever
from nexus_llm.rag.indexer import Indexer
from nexus_llm.rag.chunker import Chunker
from nexus_llm.rag.pipeline import RAGPipeline, RAGResult
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Convenient wrapper around :class:`RAGResult` for the engine API.

    Attributes:
        answer: The generated answer text.
        sources: Source document IDs used.
        scores: Retrieval scores.
        metadata: Additional metadata.
    """

    answer: str
    sources: List[str]
    scores: List[float]
    metadata: Dict[str, Any]

    @classmethod
    def from_rag_result(cls, result: RAGResult) -> "QueryResult":
        return cls(
            answer=result.answer,
            sources=[doc.id for doc in result.sources],
            scores=result.scores,
            metadata=result.metadata,
        )


class RAGEngine:
    """High-level RAG engine.

    Provides a simple ``query`` / ``add_documents`` interface while
    internally managing the document store, retriever, indexer, chunker,
    and pipeline.

    Args:
        retrieval_strategy: Retrieval strategy (keyword, semantic, hybrid).
        chunk_size: Default chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.
        top_k: Number of documents to retrieve per query.
    """

    def __init__(
        self,
        retrieval_strategy: str = "keyword",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ) -> None:
        self.store = DocumentStore()
        self.retriever = Retriever(strategy=retrieval_strategy, top_k=top_k)
        self.indexer = Indexer()
        self.chunker = Chunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.pipeline = RAGPipeline(
            document_store=self.store,
            retriever=self.retriever,
            chunker=self.chunker,
            indexer=self.indexer,
            top_k=top_k,
        )
        self._retrieval_strategy = retrieval_strategy
        logger.info(
            "RAGEngine initialised (strategy=%s, chunk_size=%d, top_k=%d)",
            retrieval_strategy, chunk_size, top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> QueryResult:
        """Run a RAG query and return a :class:`QueryResult`.

        Args:
            question: The user's question.

        Returns:
            A :class:`QueryResult` with the answer and metadata.
        """
        result = self.pipeline.run(question)
        logger.info("Query answered: %s", result.summary())
        return QueryResult.from_rag_result(result)

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the engine.

        Each dict should have a ``"content"`` key and optionally
        ``"metadata"`` and ``"id"`` keys.

        Args:
            docs: List of document dictionaries.

        Returns:
            List of assigned document IDs.
        """
        documents = [
            Document(
                id=d.get("id", ""),
                content=d["content"],
                metadata=d.get("metadata", {}),
            )
            for d in docs
        ]
        return self.pipeline.add_documents(documents)
