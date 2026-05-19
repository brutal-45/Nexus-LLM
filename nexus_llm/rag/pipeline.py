"""End-to-end RAG pipeline: retrieve, augment, and generate.

Provides a configurable pipeline that orchestrates document retrieval,
query expansion, context window management, reranking, and generation
to produce grounded answers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from nexus_llm.rag.chunker import TextChunker, SentenceChunker
from nexus_llm.rag.embeddings import EmbeddingModel
from nexus_llm.rag.indexer import DocumentIndexer
from nexus_llm.rag.reranker import Reranker, RelevanceReranker
from nexus_llm.rag.retriever import Retriever, RetrievalResult, HybridRetriever, SimilarityRetriever, BM25Retriever
from nexus_llm.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Retrieval settings
    top_k: int = 10
    rerank_top_k: int = 5
    use_reranking: bool = True
    use_hybrid_search: bool = True
    bm25_weight: float = 0.4
    similarity_weight: float = 0.6
    fusion_method: str = "rrf"

    # Query expansion
    use_query_expansion: bool = True
    max_expanded_queries: int = 3

    # Context window
    max_context_tokens: int = 4096
    context_overlap_tokens: int = 100

    # Generation settings
    generation_template: str = (
        "Based on the following context, answer the question. "
        "If the answer is not contained in the context, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    system_prompt: str = "You are a helpful assistant that answers questions based on the provided context."

    # Chunking defaults
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    query: str
    context_used: List[str] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return f"RAGResponse(query='{self.query[:40]}...', answer='{preview}...', sources={len(self.sources)})"


class QueryExpander:
    """Expands queries to improve retrieval recall.

    Generates alternative query formulations using rule-based
    and LLM-based expansion strategies.
    """

    def __init__(self, llm_fn: Optional[Callable] = None):
        self.llm_fn = llm_fn

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand a query into alternative formulations.

        Args:
            query: The original query.
            max_expansions: Maximum number of expanded queries.

        Returns:
            List of expanded query strings (includes original).
        """
        expansions = [query]

        # Rule-based expansions
        # 1. Remove question words
        stripped = re.sub(r"^(what|who|where|when|why|how|which|is|are|do|does|can|could|would|should)\s+", "", query, flags=re.IGNORECASE)
        if stripped != query:
            expansions.append(stripped.strip())

        # 2. Extract key noun phrases (simple heuristic)
        key_terms = self._extract_key_terms(query)
        if key_terms and key_terms != query:
            expansions.append(key_terms)

        # 3. LLM-based expansion if available
        if self.llm_fn and len(expansions) < max_expansions + 1:
            try:
                llm_expansions = self._llm_expand(query, max_expansions - len(expansions) + 1)
                expansions.extend(llm_expansions)
            except Exception as e:
                logger.warning("LLM query expansion failed: %s", e)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in expansions:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)

        return unique[: max_expansions + 1]

    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms from a query using simple heuristics."""
        # Remove stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "but", "and", "or", "if", "while", "about", "up", "it",
            "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
            "his", "she", "her", "they", "them", "their", "what", "which", "who",
            "whom", "this", "that", "these", "those",
        }
        tokens = query.lower().split()
        key_terms = [t for t in tokens if t not in stop_words and len(t) > 1]
        return " ".join(key_terms)

    def _llm_expand(self, query: str, count: int) -> List[str]:
        """Use LLM to generate alternative queries."""
        prompt = (
            f"Generate {count} alternative search queries that would find "
            f"the same information as: '{query}'\n"
            f"Return only the queries, one per line."
        )
        response = self.llm_fn(prompt)
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines[:count]


class ContextWindowManager:
    """Manages the context window for RAG generation.

    Selects and orders retrieved context to fit within a
    maximum token budget, ensuring the most relevant information
    is included first.
    """

    def __init__(self, max_tokens: int = 4096, overlap_tokens: int = 100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough: 1 token ≈ 4 characters)."""
        return len(text) // 4

    def build_context(
        self,
        results: List[RetrievalResult],
        query: str = "",
        separator: str = "\n\n---\n\n",
    ) -> tuple:
        """Build context string from retrieval results within token budget.

        Args:
            results: Ranked retrieval results.
            query: The original query (reserved for future use).
            separator: Separator between context chunks.

        Returns:
            Tuple of (context_string, included_results, total_tokens).
        """
        context_parts: List[str] = []
        included_results: List[RetrievalResult] = []
        total_tokens = 0
        sep_tokens = self._estimate_tokens(separator)

        for result in results:
            part = result.document.text
            part_tokens = self._estimate_tokens(part)

            if total_tokens + part_tokens + (sep_tokens if context_parts else 0) > self.max_tokens:
                # Try to fit a truncated version if no context yet
                if not context_parts:
                    remaining = self.max_tokens - total_tokens
                    char_limit = remaining * 4
                    if char_limit > 100:
                        truncated = part[:char_limit] + "..."
                        context_parts.append(truncated)
                        included_results.append(result)
                        total_tokens += self._estimate_tokens(truncated)
                break

            context_parts.append(part)
            included_results.append(result)
            total_tokens += part_tokens + (sep_tokens if len(context_parts) > 1 else 0)

        context = separator.join(context_parts)
        return context, included_results, total_tokens


class RAGPipeline:
    """End-to-end RAG pipeline orchestrating retrieval, augmentation, and generation.

    Coordinates the full RAG workflow: query expansion, hybrid retrieval,
    optional reranking, context window management, and LLM generation
    to produce grounded, source-attributed answers.
    """

    def __init__(
        self,
        indexer: DocumentIndexer,
        retriever: Retriever,
        reranker: Optional[Reranker] = None,
        llm_fn: Optional[Callable] = None,
        config: Optional[RAGConfig] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            indexer: Document indexer for adding new documents.
            retriever: Document retriever for search.
            reranker: Optional reranker for result refinement.
            llm_fn: Optional LLM generation function. Takes a prompt string,
                   returns a response string.
            config: Pipeline configuration.
        """
        self.indexer = indexer
        self.retriever = retriever
        self.reranker = reranker or RelevanceReranker()
        self.llm_fn = llm_fn
        self.config = config or RAGConfig()

        self.query_expander = QueryExpander(llm_fn=llm_fn)
        self.context_manager = ContextWindowManager(
            max_tokens=self.config.max_context_tokens,
            overlap_tokens=self.config.context_overlap_tokens,
        )

    def add_documents(self, documents: List[dict]) -> None:
        """Add documents to the RAG pipeline index.

        Args:
            documents: List of dicts with 'content', 'source', etc.
        """
        self.indexer.index_documents(documents)
        # Re-index retriever if needed
        if hasattr(self.retriever, "index_documents"):
            from nexus_llm.rag.vector_store import VectorDocument
            vdocs = []
            for record in self.indexer.list_documents():
                for chunk_id in record.chunk_ids:
                    vdoc = self.indexer.vector_store._documents.get(chunk_id)
                    if vdoc:
                        vdocs.append(vdoc)
            if vdocs and isinstance(self.retriever, (HybridRetriever,)):
                self.retriever.index_documents(vdocs)

    def query(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """Execute the full RAG pipeline for a query.

        Args:
            query: The user query.
            top_k: Override for number of results to retrieve.

        Returns:
            RAGResponse with answer, context, and sources.
        """
        top_k = top_k or self.config.top_k
        rerank_top_k = self.config.rerank_top_k

        # Step 1: Query expansion
        expanded_queries = [query]
        if self.config.use_query_expansion:
            expanded_queries = self.query_expander.expand(
                query, max_expansions=self.config.max_expanded_queries
            )
            logger.info("Expanded queries: %s", expanded_queries)

        # Step 2: Retrieve documents for all expanded queries
        all_results: Dict[str, RetrievalResult] = {}

        for q in expanded_queries:
            results = self.retriever.retrieve(q, top_k=top_k)
            for result in results:
                doc_id = result.document.doc_id
                if doc_id not in all_results or result.score > all_results[doc_id].score:
                    all_results[doc_id] = result

        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)

        # Step 3: Rerank
        if self.config.use_reranking and self.reranker and combined_results:
            combined_results = self.reranker.rerank(
                query, combined_results, top_k=rerank_top_k
            )
            logger.info("Reranked to %d results.", len(combined_results))

        # Step 4: Build context
        context, included_results, token_count = self.context_manager.build_context(
            combined_results, query
        )

        # Step 5: Generate answer
        answer = self._generate(query, context)

        # Step 6: Build response
        sources = [
            {
                "doc_id": r.document.doc_id,
                "text_preview": r.document.text[:100],
                "score": r.score,
                "method": r.retrieval_method,
                "metadata": r.document.metadata,
            }
            for r in included_results
        ]

        return RAGResponse(
            answer=answer,
            query=query,
            context_used=[r.document.text for r in included_results],
            sources=sources,
            retrieval_results=included_results,
            expanded_queries=expanded_queries,
            metadata={
                "total_results": len(combined_results),
                "context_tokens": token_count,
                "num_sources": len(sources),
            },
        )

    def _generate(self, query: str, context: str) -> str:
        """Generate an answer using the LLM.

        Args:
            query: The user query.
            context: The retrieved context.

        Returns:
            Generated answer string.
        """
        if not context:
            return "I don't have enough information to answer this question."

        prompt = self.config.generation_template.format(
            context=context, query=query
        )

        if self.llm_fn:
            try:
                answer = self.llm_fn(prompt)
                return answer.strip()
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                return f"Error generating answer: {e}"
        else:
            # Return context-based fallback
            return (
                f"Based on the retrieved context:\n\n{context[:1000]}\n\n"
                f"(No LLM available for generation. Showing retrieved context.)"
            )

    @classmethod
    def create(
        cls,
        embedding_model: EmbeddingModel,
        chunker: Optional[TextChunker] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        reranker: Optional[Reranker] = None,
        llm_fn: Optional[Callable] = None,
        config: Optional[RAGConfig] = None,
    ) -> "RAGPipeline":
        """Factory method to create a fully configured RAG pipeline.

        Args:
            embedding_model: The embedding model to use.
            chunker: Optional text chunker (defaults to SentenceChunker).
            vector_store: Optional vector store (creates new FAISS store).
            reranker: Optional reranker.
            llm_fn: Optional LLM function.
            config: Optional configuration.

        Returns:
            Configured RAGPipeline instance.
        """
        config = config or RAGConfig()

        chunker = chunker or SentenceChunker(max_chunk_size=config.chunk_size)

        vector_store = vector_store or FAISSVectorStore(
            dimension=embedding_model.dimension,
            metric="cosine",
        )

        indexer = DocumentIndexer(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Build retriever
        if config.use_hybrid_search:
            sim_retriever = SimilarityRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
            )
            bm25_retriever = BM25Retriever()
            retriever = HybridRetriever(
                retrievers=[
                    (sim_retriever, config.similarity_weight),
                    (bm25_retriever, config.bm25_weight),
                ],
                fusion_method=config.fusion_method,
            )
        else:
            retriever = SimilarityRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
            )

        reranker = reranker or RelevanceReranker()

        return cls(
            indexer=indexer,
            retriever=retriever,
            reranker=reranker,
            llm_fn=llm_fn,
            config=config,
        )
