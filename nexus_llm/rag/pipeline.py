"""RAG pipeline for Nexus-LLM.

Orchestrates the full retrieve → augment → generate workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.rag.document_store import Document, DocumentStore
from nexus_llm.rag.retriever import Retriever
from nexus_llm.rag.indexer import Indexer
from nexus_llm.rag.chunker import Chunker
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """The result of a RAG pipeline run.

    Attributes:
        query: The original query.
        answer: The generated answer (or placeholder if no generator).
        sources: Documents used to construct the answer.
        scores: Retrieval scores for each source.
        metadata: Additional pipeline metadata.
    """

    query: str
    answer: str
    sources: List[Document]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a concise summary of the result."""
        return (
            f"Query: {self.query[:80]!r} | "
            f"Sources: {len(self.sources)} | "
            f"Answer: {self.answer[:100]}..."
        )


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → augment → generate.

    Args:
        document_store: The document store to retrieve from.
        retriever: The retrieval engine.
        chunker: The chunker for splitting added documents.
        indexer: The indexer for building search indices.
        top_k: Number of documents to retrieve per query.
        generation_prompt_template: Template for the augmented prompt.
            Must contain ``{context}`` and ``{query}`` placeholders.
    """

    _DEFAULT_TEMPLATE = (
        "Based on the following context, answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\nAnswer:"
    )

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        retriever: Optional[Retriever] = None,
        chunker: Optional[Chunker] = None,
        indexer: Optional[Indexer] = None,
        top_k: int = 5,
        generation_prompt_template: Optional[str] = None,
    ) -> None:
        self.store = document_store or DocumentStore()
        self.retriever = retriever or Retriever()
        self.chunker = chunker or Chunker()
        self.indexer = indexer or Indexer()
        self.top_k = top_k
        self.template = generation_prompt_template or self._DEFAULT_TEMPLATE
        logger.info("RAGPipeline initialised (top_k=%d)", top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> RAGResult:
        """Execute the full RAG pipeline for *query*.

        Steps:
        1. **Retrieve** – fetch the top-k relevant documents.
        2. **Augment** – build a context-augmented prompt.
        3. **Generate** – produce an answer (mock if no LLM is available).

        Returns:
            A :class:`RAGResult` containing the answer and sources.
        """
        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        sources = [doc for doc, _ in retrieved]
        scores = [score for _, score in retrieved]

        if not sources:
            logger.warning("No documents retrieved for query: %s", query[:80])
            return RAGResult(
                query=query,
                answer="I could not find any relevant information to answer your question.",
                sources=[],
                scores=[],
                metadata={"retrieved": 0},
            )

        # Step 2: Augment
        context = self._build_context(sources)
        augmented_prompt = self.template.format(
            context=context, query=query,
        )

        # Step 3: Generate (mock)
        answer = self._mock_generate(augmented_prompt, sources)

        logger.info(
            "RAG pipeline complete: %d source(s), answer length=%d",
            len(sources), len(answer),
        )
        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            scores=scores,
            metadata={
                "retrieved": len(sources),
                "context_length": len(context),
                "prompt_length": len(augmented_prompt),
            },
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store, chunk them, and re-index.

        Returns:
            A list of document IDs.
        """
        doc_ids: List[str] = []
        all_chunks: List[Document] = []

        for doc in documents:
            doc_id = self.store.add(doc)
            doc_ids.append(doc_id)

            chunks = self.chunker.chunk(doc.content)
            for chunk in chunks:
                chunk_doc = Document(
                    content=chunk.content,
                    metadata={
                        "parent_id": doc_id,
                        "chunk_index": chunk.index,
                        "strategy": chunk.metadata.get("strategy") if chunk.metadata else None,
                    },
                )
                self.store.add(chunk_doc)
                all_chunks.append(chunk_doc)

        # Re-index after adding all documents
        self.indexer.build_index(self.store)
        self.retriever.index_documents(
            [self.store.get(did) for did in self.store.list_ids()
             if self.store.get(did) is not None]
        )

        logger.info("Added %d documents (%d chunks)", len(doc_ids), len(all_chunks))
        return doc_ids

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_context(self, sources: List[Document]) -> str:
        """Concatenate source documents into a context string."""
        parts: List[str] = []
        for i, doc in enumerate(sources, 1):
            title = doc.metadata.get("title", f"Source {i}")
            parts.append(f"[{title}]\n{doc.content}")
        return "\n\n---\n\n".join(parts)

    def _mock_generate(self, prompt: str, sources: List[Document]) -> str:
        """Mock generation when no LLM is connected.

        Returns a summarised answer based on the source documents.
        """
        if not sources:
            return "No relevant information found."

        # Simple extractive approach: use the top source as the answer
        top_source = sources[0]
        answer = top_source.content

        # If we have multiple sources, add a reference note
        if len(sources) > 1:
            answer += (
                f"\n\n[Based on {len(sources)} source document(s)]"
            )

        return answer
