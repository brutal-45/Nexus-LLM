"""
Nexus LLM — Knowledge Configuration
=====================================

Comprehensive dataclass-based configuration for every knowledge subsystem:
retrieval, embeddings, vector stores, RAG pipelines, knowledge graphs,
and document chunking.  Every field carries a sensible default so that
instantiating a config class with zero arguments gives a fully usable
baseline.

Design principles
-----------------
* Immutable via ``frozen=True`` dataclasses where appropriate.
* Validation on construction via ``__post_init__``.
* ``to_dict()`` / ``from_dict()`` round-trip for serialisation.
* Typed with :mod:`typing` literals for IDE auto-complete.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

logger: logging.Logger = logging.getLogger("nexus.knowledge.config")


# ═══════════════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalMethod(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    COLBERT = "colbert"

class PoolingStrategy(str, Enum):
    MEAN = "mean"
    CLS = "cls"
    MAX = "max"

class VectorStoreBackend(str, Enum):
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    CHROMADB = "chromadb"
    SIMPLE = "simple"

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "ip"

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"

class CitationStyle(str, Enum):
    INLINE_NUMERIC = "inline_numeric"
    INLINE_BRACKET = "inline_bracket"
    FOOTNOTE = "footnote"
    NONE = "none"

class RerankerType(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    T5 = "t5"
    COLBERT = "colbert"
    MMR = "mmr"
    DISTILLED = "distilled"
    LISTWISE = "listwise"
    ENSEMBLE = "ensemble"


# ═══════════════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clamp(value: int, lo: int, hi: int, name: str) -> int:
    """Clamp *value* to ``[lo, hi]`` and warn if it was out of range."""
    if value < lo:
        logger.warning("%s=%d is below minimum %d; clamping.", name, value, lo)
        return lo
    if value > hi:
        logger.warning("%s=%d is above maximum %d; clamping.", name, value, hi)
        return hi
    return value


def _clamp_float(value: float, lo: float, hi: float, name: str) -> float:
    if value < lo:
        logger.warning("%s=%.4f is below minimum %.4f; clamping.", name, value, lo)
        return lo
    if value > hi:
        logger.warning("%s=%.4f is above maximum %.4f; clamping.", name, value, hi)
        return hi
    return value


def _validate_positive(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _validate_non_negative(value: float, name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _validate_probability(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


# ═══════════════════════════════════════════════════════════════════════════
#  Base serialisation mixin
# ═══════════════════════════════════════════════════════════════════════════

class _ConfigMixin:
    """Shared ``to_dict`` / ``from_dict`` / ``to_json`` helpers."""

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialise this config to a plain ``dict``."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value) and not isinstance(value, type):
                out[f.name] = value.to_dict()  # type: ignore[union-attr]
            elif isinstance(value, Enum):
                out[f.name] = value.value
            elif isinstance(value, (list, tuple)):
                out[f.name] = [
                    v.to_dict() if is_dataclass(v) and not isinstance(v, type) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                out[f.name] = {
                    k: v.to_dict() if is_dataclass(v) and not isinstance(v, type) else v
                    for k, v in value.items()
                }
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_ConfigMixin":
        """Create an instance from a plain ``dict`` (inverse of ``to_dict``)."""
        import typing

        field_types: Dict[str, Any] = {f.name: f.type for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in field_types:
                logger.debug("Ignoring unknown key %r in config %s", key, cls.__name__)
                continue
            ftype = field_types[key]
            origin = getattr(ftype, "__origin__", None)
            if origin is Union:
                args = getattr(ftype, "__args__", ())
                for arg in args:
                    if is_dataclass(arg) and isinstance(value, dict):
                        value = arg.from_dict(value)
                        break
            elif is_dataclass(ftype) and isinstance(value, dict):
                value = ftype.from_dict(value)
            elif (
                origin is list
                and hasattr(ftype, "__args__")
                and len(ftype.__args__) > 0
            ):
                inner = ftype.__args__[0]
                if is_dataclass(inner) and isinstance(value, list):
                    value = [inner.from_dict(v) if isinstance(v, dict) else v for v in value]
            kwargs[key] = value
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "_ConfigMixin":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def save_json(self, path: Union[str, Path], indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")

    def copy(self) -> _ConfigMixin:
        return copy.deepcopy(self)


# ═══════════════════════════════════════════════════════════════════════════
#  RetrievalConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalConfig(_ConfigMixin):
    """Configuration for the retrieval subsystem.

    Attributes
    ----------
    method : RetrievalMethod
        Retrieval paradigm — ``dense``, ``sparse``, ``hybrid``, or ``colbert``.
    top_k : int
        Number of results to return per query.
    score_threshold : float
        Minimum relevance score (0-1) to include a result.  Set to ``0.0``
        to disable thresholding.
    reranker : Optional[RerankerType]
        Post-retrieval reranker to apply.  ``None`` disables reranking.
    max_docs : int
        Maximum number of documents to retrieve before reranking.
    bm25_k1 : float
        BM25 term-saturation parameter (only used for sparse retrieval).
    bm25_b : float
        BM25 length-normalisation parameter (only used for sparse retrieval).
    bm25_epsilon : float
        BM25 lower-bound for IDF to prevent negative values.
    hybrid_alpha : float
        Weight for dense scores in hybrid retrieval (``1 - alpha`` goes to
        sparse).  Only used when ``method == HYBRID``.
    rrf_k : int
        Reciprocal-rank-fusion constant.
    colbert_dim : int
        Dimensionality for ColBERT late-interaction token embeddings.
    colbert_score_scale : float
        Temperature scaling applied to ColBERT MaxSim scores.
    batch_size : int
        Batch size for encoding operations.
    device : str
        Compute device string, e.g. ``"cpu"``, ``"cuda:0"``.
    """

    method: RetrievalMethod = RetrievalMethod.DENSE
    top_k: int = 10
    score_threshold: float = 0.0
    reranker: Optional[RerankerType] = None
    max_docs: int = 100
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    hybrid_alpha: float = 0.7
    rrf_k: int = 60
    colbert_dim: int = 128
    colbert_score_scale: float = 1.0
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.top_k = _validate_positive(self.top_k, "top_k")
        self.max_docs = _validate_positive(self.max_docs, "max_docs")
        self.score_threshold = _validate_probability(self.score_threshold, "score_threshold")
        self.bm25_k1 = _clamp_float(self.bm25_k1, 0.0, 3.0, "bm25_k1")
        self.bm25_b = _clamp_float(self.bm25_b, 0.0, 1.0, "bm25_b")
        self.bm25_epsilon = _clamp_float(self.bm25_epsilon, 0.01, 1.0, "bm25_epsilon")
        self.hybrid_alpha = _validate_probability(self.hybrid_alpha, "hybrid_alpha")
        self.rrf_k = _validate_positive(self.rrf_k, "rrf_k")
        self.colbert_dim = _validate_positive(self.colbert_dim, "colbert_dim")
        self.colbert_score_scale = _clamp_float(self.colbert_score_scale, 0.01, 10.0, "colbert_score_scale")
        self.batch_size = _validate_positive(self.batch_size, "batch_size")


# ═══════════════════════════════════════════════════════════════════════════
#  EmbeddingConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingConfig(_ConfigMixin):
    """Configuration for the embedding model.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier or ``"simple"`` for the built-in
        embedding model.
    dimension : int
        Output embedding dimensionality.
    pooling : PoolingStrategy
        How to pool token-level representations into a fixed-size vector.
    normalize : bool
        L2-normalise embeddings after pooling.
    batch_size : int
        Number of texts to encode per batch.
    max_seq_length : int
        Maximum sequence length for the tokenizer.
    query_prefix : str
        Prefix prepended to queries before encoding (E5/BGE convention).
    document_prefix : str
        Prefix prepended to documents before encoding.
    hidden_size : int
        Hidden dimension for the built-in ``SimpleEmbedding`` model.
    num_layers : int
        Number of transformer layers for ``SimpleEmbedding``.
    num_heads : int
        Number of attention heads for ``SimpleEmbedding``.
    dropout : float
        Dropout probability for the embedding model.
    freeze_layers : int
        Number of bottom layers to freeze during fine-tuning (0 = none).
    matryoshka_dims : Optional[List[int]]
        Nested dimension targets for Matryoshka embeddings, e.g.
        ``[64, 128, 256, 512, 768]``.
    device : str
        Compute device.
    dtype : str
        Floating-point precision (``"float32"`` or ``"float16"``).
    """

    model_name: str = "simple"
    dimension: int = 768
    pooling: PoolingStrategy = PoolingStrategy.MEAN
    normalize: bool = True
    batch_size: int = 32
    max_seq_length: int = 512
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    dropout: float = 0.1
    freeze_layers: int = 0
    matryoshka_dims: Optional[List[int]] = None
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.dimension = _validate_positive(self.dimension, "dimension")
        self.batch_size = _validate_positive(self.batch_size, "batch_size")
        self.max_seq_length = _clamp(self.max_seq_length, 32, 32768, "max_seq_length")
        self.hidden_size = _validate_positive(self.hidden_size, "hidden_size")
        self.num_layers = _validate_positive(self.num_layers, "num_layers")
        self.num_heads = _validate_positive(self.num_heads, "num_heads")
        self.dropout = _validate_probability(self.dropout, "dropout")
        self.freeze_layers = _validate_non_negative(float(self.freeze_layers), "freeze_layers")
        self.freeze_layers = int(self.freeze_layers)
        if self.matryoshka_dims is not None:
            if not self.matryoshka_dims:
                raise ValueError("matryoshka_dims must be non-empty")
            if self.dimension not in self.matryoshka_dims:
                raise ValueError(
                    f"dimension ({self.dimension}) must be in matryoshka_dims "
                    f"({self.matryoshka_dims})"
                )
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"dtype must be float32/float16/bfloat16, got {self.dtype}")


# ═══════════════════════════════════════════════════════════════════════════
#  VectorStoreConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VectorStoreConfig(_ConfigMixin):
    """Configuration for the vector store backend.

    Attributes
    ----------
    backend : VectorStoreBackend
        Which vector store implementation to use.
    index_type : str
        Type of index within the backend (``"flat"``, ``"hnsw"``, ``"ivf"``,
        ``"ivf_pq"``).
    metric : DistanceMetric
        Distance metric for similarity computation.
    nlist : int
        Number of Voronoi cells for IVF-type indices.
    m : int
        Number of neighbours per layer for HNSW.
    ef_construction : int
        Construction-time beam width for HNSW.
    ef_search : int
        Query-time beam width for HNSW.
    nprobe : int
        Number of IVF cells to probe at query time.
    pq_subquantizers : int
        Number of product-quantization sub-quantizers.
    pq_bits : int
        Bits per sub-quantizer code.
    pq_niter : int
        K-means iterations when training PQ codebooks.
    use_disk : bool
        Enable disk-based indexing (DiskANN mode).
    disk_page_size : int
        Page size in bytes for the disk index.
    disk_max_degree : int
        Maximum graph degree for DiskANN.
    disk_search_beam_width : int
        Beam width for disk-based search.
    persist_path : Optional[str]
        Directory where the index is persisted.  ``None`` = in-memory only.
    read_only : bool
        Open the store in read-only mode.
    """

    backend: VectorStoreBackend = VectorStoreBackend.SIMPLE
    index_type: str = "flat"
    metric: DistanceMetric = DistanceMetric.COSINE
    nlist: int = 100
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    nprobe: int = 10
    pq_subquantizers: int = 8
    pq_bits: int = 8
    pq_niter: int = 20
    use_disk: bool = False
    disk_page_size: int = 4096
    disk_max_degree: int = 64
    disk_search_beam_width: int = 16
    persist_path: Optional[str] = None
    read_only: bool = False

    def __post_init__(self) -> None:
        self.nlist = _validate_positive(self.nlist, "nlist")
        self.m = _clamp(self.m, 2, 128, "m")
        self.ef_construction = _validate_positive(self.ef_construction, "ef_construction")
        self.ef_search = _validate_positive(self.ef_search, "ef_search")
        self.nprobe = _clamp(self.nprobe, 1, self.nlist, "nprobe")
        self.pq_subquantizers = _clamp(self.pq_subquantizers, 1, 256, "pq_subquantizers")
        self.pq_bits = _clamp(self.pq_bits, 1, 16, "pq_bits")
        self.pq_niter = _validate_positive(self.pq_niter, "pq_niter")
        self.disk_page_size = _validate_positive(self.disk_page_size, "disk_page_size")
        self.disk_max_degree = _clamp(self.disk_max_degree, 4, 256, "disk_max_degree")
        self.disk_search_beam_width = _validate_positive(self.disk_search_beam_width, "disk_search_beam_width")


# ═══════════════════════════════════════════════════════════════════════════
#  RAGConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RAGConfig(_ConfigMixin):
    """End-to-end RAG pipeline configuration.

    Attributes
    ----------
    retriever : RetrievalConfig
        Retrieval sub-config.
    reranker : RerankerType
        Reranker to use after initial retrieval.
    generator_model : str
        LLM model identifier for answer generation.
    max_context_tokens : int
        Maximum number of tokens in the assembled context window.
    num_retrieved_docs : int
        Number of documents retrieved per step.
    citation_style : CitationStyle
        How to cite sources in generated answers.
    citation_prefix : str
        Prefix for inline citation markers.
    context_separator : str
        Separator placed between document chunks in the context.
    include_metadata : bool
        Whether to prepend document metadata to each context chunk.
    system_prompt : str
        System prompt that instructs the generator.
    query_rewrite : bool
        Whether to rewrite user queries before retrieval.
    query_expansion_count : int
        Number of synthetic expansion queries (0 = disabled).
    multi_step : bool
        Enable iterative retrieve-reason-retrieve cycles.
    multi_step_max_iterations : int
        Maximum number of multi-step iterations.
    faithfulness_check : bool
        Post-generation faithfulness check against retrieved context.
    faithfulness_threshold : float
        Minimum faithfulness score (0-1).
    streaming : bool
        Whether to stream generator output.
    temperature : float
        Sampling temperature for the generator.
    max_new_tokens : int
        Maximum tokens to generate per answer.
    """

    retriever: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerType = RerankerType.CROSS_ENCODER
    generator_model: str = "simple"
    max_context_tokens: int = 4096
    num_retrieved_docs: int = 5
    citation_style: CitationStyle = CitationStyle.INLINE_NUMERIC
    citation_prefix: str = "["
    context_separator: str = "\n\n---\n\n"
    include_metadata: bool = True
    system_prompt: str = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain enough information, "
        "say so honestly. Cite your sources."
    )
    query_rewrite: bool = True
    query_expansion_count: int = 0
    multi_step: bool = False
    multi_step_max_iterations: int = 3
    faithfulness_check: bool = False
    faithfulness_threshold: float = 0.7
    streaming: bool = False
    temperature: float = 0.1
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        self.max_context_tokens = _clamp(self.max_context_tokens, 128, 131072, "max_context_tokens")
        self.num_retrieved_docs = _validate_positive(self.num_retrieved_docs, "num_retrieved_docs")
        self.query_expansion_count = _clamp(self.query_expansion_count, 0, 10, "query_expansion_count")
        self.multi_step_max_iterations = _clamp(self.multi_step_max_iterations, 1, 10, "multi_step_max_iterations")
        self.faithfulness_threshold = _validate_probability(self.faithfulness_threshold, "faithfulness_threshold")
        self.temperature = _clamp_float(self.temperature, 0.0, 2.0, "temperature")
        self.max_new_tokens = _validate_positive(self.max_new_tokens, "max_new_tokens")


# ═══════════════════════════════════════════════════════════════════════════
#  KnowledgeGraphConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeGraphConfig(_ConfigMixin):
    """Configuration for the knowledge-graph subsystem.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of node/relation embeddings.
    max_entities : int
        Maximum number of entities before eviction kicks in.
    max_relations : int
        Maximum number of relations.
    similarity_threshold : float
        Minimum cosine similarity for entity linking.
    embedding_model : str
        Embedding model identifier for entity/relation embeddings.
    max_triple_depth : int
        Maximum depth for graph traversals and reasoning.
    pruning_enabled : bool
        Automatically prune isolated nodes.
    pruning_threshold : int
        Minimum degree for a node to survive pruning.
    reasoning_model : str
        Model for multi-hop reasoning over the graph.
    merge_strategy : str
        How to merge overlapping entities (``"union"``, ``"intersect"``,
        ``"weighted"``).
    persist_path : Optional[str]
        Directory for graph persistence.
    auto_entity_extraction : bool
        Automatically extract entities from incoming text.
    auto_relation_extraction : bool
        Automatically extract relations from incoming text.
    """

    embedding_dim: int = 256
    max_entities: int = 100_000
    max_relations: int = 500_000
    similarity_threshold: float = 0.85
    embedding_model: str = "simple"
    max_triple_depth: int = 3
    pruning_enabled: bool = True
    pruning_threshold: int = 1
    reasoning_model: str = "simple"
    merge_strategy: str = "weighted"
    persist_path: Optional[str] = None
    auto_entity_extraction: bool = True
    auto_relation_extraction: bool = True

    def __post_init__(self) -> None:
        self.embedding_dim = _validate_positive(self.embedding_dim, "embedding_dim")
        self.max_entities = _validate_positive(self.max_entities, "max_entities")
        self.max_relations = _validate_positive(self.max_relations, "max_relations")
        self.similarity_threshold = _validate_probability(self.similarity_threshold, "similarity_threshold")
        self.max_triple_depth = _clamp(self.max_triple_depth, 1, 10, "max_triple_depth")
        self.pruning_threshold = _clamp(self.pruning_threshold, 0, 100, "pruning_threshold")
        if self.merge_strategy not in ("union", "intersect", "weighted"):
            raise ValueError(f"merge_strategy must be union/intersect/weighted, got {self.merge_strategy}")


# ═══════════════════════════════════════════════════════════════════════════
#  ChunkingConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkingConfig(_ConfigMixin):
    """Configuration for document chunking.

    Attributes
    ----------
    strategy : ChunkingStrategy
        Chunking method to apply.
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    min_chunk_size : int
        Discard chunks shorter than this (characters).
    max_chunk_size : int
        Hard cap on chunk length (characters).
    sentence_threshold : float
        Cosine similarity threshold for semantic chunking.
    separators : List[str]
        Ordered list of separators for recursive splitting (tried in order).
    respect_sentence_boundary : bool
        Never split in the middle of a sentence.
    strip_whitespace : bool
        Strip leading/trailing whitespace from each chunk.
    add_chunk_index : bool
        Prepend the zero-based chunk index as metadata.
    semantic_batch_size : int
        Batch size for embedding during semantic chunking.
    semantic_window_size : int
        Window size for similarity comparison in semantic chunking.
    sliding_window_step : int
        Step size for the sliding-window chunker (smaller ⇒ more overlap).
    heading_levels : List[int]
        Markdown heading levels to split on (e.g. ``[1, 2, 3]``).
    agentic_llm_model : str
        Model name for the LLM-based agentic chunker.
    """

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 64
    max_chunk_size: int = 2048
    sentence_threshold: float = 0.5
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    respect_sentence_boundary: bool = True
    strip_whitespace: bool = True
    add_chunk_index: bool = True
    semantic_batch_size: int = 32
    semantic_window_size: int = 2
    sliding_window_step: int = 256
    heading_levels: List[int] = field(default_factory=lambda: [1, 2, 3])
    agentic_llm_model: str = "simple"

    def __post_init__(self) -> None:
        self.chunk_size = _validate_positive(self.chunk_size, "chunk_size")
        self.chunk_overlap = _clamp(self.chunk_overlap, 0, self.chunk_size - 1, "chunk_overlap")
        self.min_chunk_size = _clamp(self.min_chunk_size, 1, self.chunk_size, "min_chunk_size")
        self.max_chunk_size = _validate_positive(self.max_chunk_size, "max_chunk_size")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")
        self.sentence_threshold = _validate_probability(self.sentence_threshold, "sentence_threshold")
        self.semantic_batch_size = _validate_positive(self.semantic_batch_size, "semantic_batch_size")
        self.semantic_window_size = _validate_positive(self.semantic_window_size, "semantic_window_size")
        self.sliding_window_step = _validate_positive(self.sliding_window_step, "sliding_window_step")
        if self.separators is None or len(self.separators) == 0:
            raise ValueError("separators must be a non-empty list")


# ═══════════════════════════════════════════════════════════════════════════
#  Full KnowledgeConfig (aggregates everything)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeConfig(_ConfigMixin):
    """Top-level configuration that aggregates all knowledge sub-configs.

    This is the single entry-point for configuring the entire knowledge
    module.  Individual sub-configs can be overridden after construction.

    Attributes
    ----------
    retrieval : RetrievalConfig
    embedding : EmbeddingConfig
    vector_store : VectorStoreConfig
    rag : RAGConfig
    knowledge_graph : KnowledgeGraphConfig
    chunking : ChunkingConfig
    """

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration resolver / factory
# ═══════════════════════════════════════════════════════════════════════════

class ConfigResolver:
    """Resolve a :class:`KnowledgeConfig` into concrete component instances.

    The resolver inspects each sub-config's fields and wires together
    compatible components — e.g. selecting a dense retriever that shares
    the embedding model from the embedding config.
    """

    def __init__(self, config: Optional[KnowledgeConfig] = None) -> None:
        self._config: KnowledgeConfig = config or KnowledgeConfig()

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def set_config(self, config: KnowledgeConfig) -> None:
        self._config = config

    # ── Retrieval ──────────────────────────────────────────────────────

    def get_retrieval_config(self) -> RetrievalConfig:
        return self._config.retrieval

    def get_dense_retriever_kwargs(self) -> Dict[str, Any]:
        cfg = self._config.retrieval
        emb = self._config.embedding
        return {
            "dimension": emb.dimension,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "top_k": cfg.top_k,
            "score_threshold": cfg.score_threshold,
            "max_docs": cfg.max_docs,
        }

    def get_sparse_retriever_kwargs(self) -> Dict[str, Any]:
        cfg = self._config.retrieval
        return {
            "k1": cfg.bm25_k1,
            "b": cfg.bm25_b,
            "epsilon": cfg.bm25_epsilon,
            "top_k": cfg.top_k,
        }

    def get_hybrid_retriever_kwargs(self) -> Dict[str, Any]:
        kwargs = self.get_dense_retriever_kwargs()
        kwargs["sparse_kwargs"] = self.get_sparse_retriever_kwargs()
        kwargs["alpha"] = self._config.retrieval.hybrid_alpha
        kwargs["rrf_k"] = self._config.retrieval.rrf_k
        return kwargs

    # ── Embedding ──────────────────────────────────────────────────────

    def get_embedding_config(self) -> EmbeddingConfig:
        return self._config.embedding

    # ── Vector Store ───────────────────────────────────────────────────

    def get_vector_store_config(self) -> VectorStoreConfig:
        return self._config.vector_store

    def get_vector_store_kwargs(self) -> Dict[str, Any]:
        cfg = self._config.vector_store
        emb = self._config.embedding
        return {
            "dimension": emb.dimension,
            "metric": cfg.metric.value,
            "backend": cfg.backend.value,
            "index_type": cfg.index_type,
            "nlist": cfg.nlist,
            "m": cfg.m,
            "ef_construction": cfg.ef_construction,
            "ef_search": cfg.ef_search,
            "nprobe": cfg.nprobe,
            "persist_path": cfg.persist_path,
            "read_only": cfg.read_only,
        }

    # ── RAG ────────────────────────────────────────────────────────────

    def get_rag_config(self) -> RAGConfig:
        return self._config.rag

    # ── Knowledge Graph ────────────────────────────────────────────────

    def get_knowledge_graph_config(self) -> KnowledgeGraphConfig:
        return self._config.knowledge_graph

    # ── Chunking ───────────────────────────────────────────────────────

    def get_chunking_config(self) -> ChunkingConfig:
        return self._config.chunking

    # ── Unified summary ────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the active configuration."""
        lines: List[str] = ["=" * 60, "Nexus Knowledge — Configuration Summary", "=" * 60]
        for sub_name, sub_cfg in (
            ("Retrieval", self._config.retrieval),
            ("Embedding", self._config.embedding),
            ("Vector Store", self._config.vector_store),
            ("RAG", self._config.rag),
            ("Knowledge Graph", self._config.knowledge_graph),
            ("Chunking", self._config.chunking),
        ):
            lines.append(f"\n--- {sub_name} ---")
            for f in fields(sub_cfg):
                val = getattr(sub_cfg, f.name)
                if is_dataclass(val) and not isinstance(val, type):
                    val = "<nested config>"
                elif isinstance(val, list) and len(val) > 5:
                    val = f"[{len(val)} items]"
                lines.append(f"  {f.name:.<30s} {val}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration presets
# ═══════════════════════════════════════════════════════════════════════════

def fast_config() -> KnowledgeConfig:
    """Minimal config for fast prototyping / unit tests."""
    return KnowledgeConfig(
        retrieval=RetrievalConfig(
            method=RetrievalMethod.DENSE,
            top_k=5,
            max_docs=50,
            batch_size=16,
        ),
        embedding=EmbeddingConfig(
            model_name="simple",
            dimension=256,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            pooling=PoolingStrategy.MEAN,
            normalize=True,
            batch_size=16,
            max_seq_length=128,
        ),
        vector_store=VectorStoreConfig(
            backend=VectorStoreBackend.SIMPLE,
            metric=DistanceMetric.COSINE,
        ),
        rag=RAGConfig(
            num_retrieved_docs=3,
            max_context_tokens=1024,
            faithfulness_check=False,
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=256,
            chunk_overlap=32,
        ),
    )


def production_config() -> KnowledgeConfig:
    """Recommended config for production workloads."""
    return KnowledgeConfig(
        retrieval=RetrievalConfig(
            method=RetrievalMethod.HYBRID,
            top_k=20,
            score_threshold=0.3,
            reranker=RerankerType.CROSS_ENCODER,
            max_docs=200,
            batch_size=64,
        ),
        embedding=EmbeddingConfig(
            dimension=768,
            pooling=PoolingStrategy.MEAN,
            normalize=True,
            batch_size=64,
            max_seq_length=512,
        ),
        vector_store=VectorStoreConfig(
            backend=VectorStoreBackend.SIMPLE,
            index_type="hnsw",
            metric=DistanceMetric.COSINE,
            m=32,
            ef_construction=256,
            ef_search=128,
        ),
        rag=RAGConfig(
            num_retrieved_docs=10,
            max_context_tokens=8192,
            reranker=RerankerType.CROSS_ENCODER,
            citation_style=CitationStyle.INLINE_BRACKET,
            query_rewrite=True,
            multi_step=True,
            multi_step_max_iterations=3,
            faithfulness_check=True,
            faithfulness_threshold=0.7,
            streaming=True,
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=512,
            chunk_overlap=64,
            sentence_threshold=0.5,
        ),
    )


def small_scale_config() -> KnowledgeConfig:
    """Config for < 10 K documents, single-machine deployment."""
    return KnowledgeConfig(
        retrieval=RetrievalConfig(
            method=RetrievalMethod.DENSE,
            top_k=10,
            score_threshold=0.2,
            batch_size=32,
        ),
        embedding=EmbeddingConfig(
            dimension=384,
            pooling=PoolingStrategy.MEAN,
            normalize=True,
            batch_size=32,
            max_seq_length=256,
        ),
        vector_store=VectorStoreConfig(
            backend=VectorStoreBackend.SIMPLE,
            index_type="flat",
            metric=DistanceMetric.COSINE,
        ),
        rag=RAGConfig(
            num_retrieved_docs=5,
            max_context_tokens=4096,
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=512,
            chunk_overlap=64,
        ),
    )


def large_scale_config() -> KnowledgeConfig:
    """Config for > 1 M documents, distributed deployment."""
    return KnowledgeConfig(
        retrieval=RetrievalConfig(
            method=RetrievalMethod.HYBRID,
            top_k=50,
            score_threshold=0.1,
            reranker=RerankerType.ENSEMBLE,
            max_docs=500,
            batch_size=128,
        ),
        embedding=EmbeddingConfig(
            dimension=1024,
            pooling=PoolingStrategy.MEAN,
            normalize=True,
            batch_size=128,
            max_seq_length=512,
        ),
        vector_store=VectorStoreConfig(
            backend=VectorStoreBackend.SIMPLE,
            index_type="ivf_pq",
            metric=DistanceMetric.INNER_PRODUCT,
            nlist=4096,
            nprobe=64,
            pq_subquantizers=32,
            pq_bits=8,
        ),
        rag=RAGConfig(
            num_retrieved_docs=20,
            max_context_tokens=16384,
            reranker=RerankerType.ENSEMBLE,
            citation_style=CitationStyle.INLINE_BRACKET,
            query_rewrite=True,
            query_expansion_count=3,
            multi_step=True,
            multi_step_max_iterations=5,
            faithfulness_check=True,
            faithfulness_threshold=0.8,
            streaming=True,
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=768,
            chunk_overlap=96,
        ),
    )
