"""
Nexus Memory Module
===================
Comprehensive memory system for LLM agents including long-term memory,
working memory, episodic memory, semantic memory, and unified management.

Architecture
------------
- **LongTermMemoryStore**: Persistent vector-indexed memory with cosine similarity
  retrieval, decay, and consolidation. Stores content as embeddings for semantic search.
- **WorkingMemoryBuffer**: Limited-capacity key-value scratchpad for active reasoning.
- **Scratchpad**: Freeform text buffer for chain-of-thought and intermediate results.
- **TaskStateTracker**: Tracks multi-step task progress with key-value state.
- **EpisodicMemoryStore**: Records and recalls past experiences (episodes) with
  emotional weighting, temporal ordering, and similarity-based recall.
- **SemanticMemoryStore**: Stores and queries factual knowledge with confidence
  scoring, contradiction detection, and fact merging.
- **MemoryManager**: Unified coordinator that orchestrates all memory types,
  handles consolidation, decay, and context retrieval across stores.

Quick Start
-----------
    from nexus.memory import MemoryManager, MemoryConfig

    config = MemoryConfig(capacity=10000, embedding_dim=256)
    manager = MemoryManager(config)

    # Store information
    manager.process_input("The user prefers concise answers", {"source": "preference"})

    # Retrieve relevant context
    context = manager.retrieve_context("What does the user prefer?")
    for entry in context:
        print(f"[{entry['source']}] {entry['content']} (score: {entry['score']:.3f})")

    # Consolidate working memory to long-term
    manager.consolidate()

Dependencies
------------
- torch (>= 1.8) — tensor operations, embedding layer, cosine similarity
- numpy (>= 1.19) — numerical utilities
- Standard library: json, hashlib, time, os, math, collections, dataclasses, typing

Version History
---------------
- 1.0.0 — Initial release with all five memory subsystems
"""

__version__ = "1.0.0"
__author__ = "Nexus LLM Team"

# ─── Configuration & Data Classes ────────────────────────────────────────────
from nexus.memory.long_term_memory import (
    MemoryConfig,
    MemoryEntry,
    MemoryEncoder,
    LongTermMemoryStore,
    MemoryConsolidator,
    MemoryDecay,
)

# ─── Working Memory ──────────────────────────────────────────────────────────
from nexus.memory.working_memory import (
    WorkingMemoryBuffer,
    Scratchpad,
    TaskStateTracker,
)

# ─── Episodic Memory ─────────────────────────────────────────────────────────
from nexus.memory.episodic_memory import (
    Episode,
    EpisodicMemoryStore,
)

# ─── Semantic Memory ──────────────────────────────────────────────────────────
from nexus.memory.semantic_memory import (
    Fact,
    SemanticMemoryStore,
)

# ─── Unified Manager ─────────────────────────────────────────────────────────
from nexus.memory.memory_manager import (
    MemoryManager,
    MemoryManagerConfig,
    RetrievalResult,
)

# ─── Public API ───────────────────────────────────────────────────────────────
__all__ = [
    # Long-term memory
    "MemoryConfig",
    "MemoryEntry",
    "MemoryEncoder",
    "LongTermMemoryStore",
    "MemoryConsolidator",
    "MemoryDecay",
    # Working memory
    "WorkingMemoryBuffer",
    "Scratchpad",
    "TaskStateTracker",
    # Episodic memory
    "Episode",
    "EpisodicMemoryStore",
    # Semantic memory
    "Fact",
    "SemanticMemoryStore",
    # Unified manager
    "MemoryManager",
    "MemoryManagerConfig",
    "RetrievalResult",
]


# ─── Factory Helpers ─────────────────────────────────────────────────────────

def create_memory_manager(
    capacity: int = 10000,
    embedding_dim: int = 256,
    persistence_path: str = None,
    decay_rate: float = 0.01,
    similarity_threshold: float = 0.7,
    working_memory_capacity: int = 64,
    scratchpad_max_length: int = 4096,
    enable_episodic: bool = True,
    enable_semantic: bool = True,
) -> MemoryManager:
    """Create a fully configured MemoryManager with sensible defaults.

    This is the recommended entry point for setting up the memory system.
    All parameters have reasonable defaults for most use cases.

    Args:
        capacity: Maximum number of long-term memories to store.
        embedding_dim: Dimensionality of embedding vectors.
        persistence_path: Filesystem path for persisting long-term memory.
            If None, memory is kept in-memory only.
        decay_rate: Rate at which memory importance decays over time.
            Higher values cause faster forgetting. Range: [0.0, 1.0].
        similarity_threshold: Minimum cosine similarity for retrieval results.
            Range: [0.0, 1.0].
        working_memory_capacity: Maximum entries in working memory buffer.
        scratchpad_max_length: Maximum character length of the scratchpad.
        enable_episodic: Whether to enable episodic (experience) memory.
        enable_semantic: Whether to enable semantic (factual) memory.

    Returns:
        A fully initialized MemoryManager instance ready for use.

    Example:
        >>> manager = create_memory_manager(
        ...     capacity=50000,
        ...     persistence_path="./memory_store",
        ...     similarity_threshold=0.6,
        ... )
        >>> manager.process_input("Important context to remember", {"source": "user"})
        >>> results = manager.retrieve_context("What context do I have?")
    """
    from nexus.memory.memory_manager import MemoryManagerConfig

    manager_config = MemoryManagerConfig(
        long_term_config=MemoryConfig(
            capacity=capacity,
            embedding_dim=embedding_dim,
            persistence_path=persistence_path,
            decay_rate=decay_rate,
            similarity_threshold=similarity_threshold,
        ),
        working_memory_capacity=working_memory_capacity,
        scratchpad_max_length=scratchpad_max_length,
        enable_episodic=enable_episodic,
        enable_semantic=enable_semantic,
    )
    return MemoryManager(manager_config)


def create_working_memory(capacity: int = 64) -> WorkingMemoryBuffer:
    """Create a standalone working memory buffer.

    Args:
        capacity: Maximum number of key-value pairs.

    Returns:
        Initialized WorkingMemoryBuffer.

    Example:
        >>> wm = create_working_memory(32)
        >>> wm.write("current_goal", "Summarize the document")
        >>> wm.read("current_goal")
        'Summarize the document'
    """
    return WorkingMemoryBuffer(capacity=capacity)


def create_scratchpad(max_length: int = 4096) -> Scratchpad:
    """Create a standalone text scratchpad for reasoning.

    Args:
        max_length: Maximum character length.

    Returns:
        Initialized Scratchpad.

    Example:
        >>> pad = create_scratchpad()
        >>> pad.append("Step 1: Read the input\\n")
        >>> pad.append("Step 2: Analyze key concepts\\n")
        >>> pad.read()
        'Step 1: Read the input\\nStep 2: Analyze key concepts\\n'
    """
    return Scratchpad(max_length=max_length)


def create_episodic_store(embedding_dim: int = 256) -> EpisodicMemoryStore:
    """Create a standalone episodic memory store.

    Args:
        embedding_dim: Dimensionality of content embeddings.

    Returns:
        Initialized EpisodicMemoryStore.

    Example:
        >>> store = create_episodic_store()
        >>> store.record_episode(
        ...     experience="User asked about Python decorators",
        ...     context="Technical support conversation",
        ...     emotional_weight=0.8,
        ...     tags=["python", "technical", "support"],
        ... )
        >>> similar = store.recall_similar("Questions about decorators", k=3)
    """
    return EpisodicMemoryStore(embedding_dim=embedding_dim)


def create_semantic_store(embedding_dim: int = 256) -> SemanticMemoryStore:
    """Create a standalone semantic (factual) memory store.

    Args:
        embedding_dim: Dimensionality of content embeddings.

    Returns:
        Initialized SemanticMemoryStore.

    Example:
        >>> store = create_semantic_store()
        >>> store.store_fact(
        ...     content="Python 3.12 was released in October 2023",
        ...     source="official_release_notes",
        ...     confidence=0.95,
        ...     category="release_info",
        ... )
        >>> results = store.query_facts("When was Python 3.12 released?", top_k=3)
    """
    return SemanticMemoryStore(embedding_dim=embedding_dim)


# ─── Version Info ────────────────────────────────────────────────────────────

def get_version() -> str:
    """Return the current version of the Nexus memory module."""
    return __version__


def get_module_info() -> dict:
    """Return detailed information about the memory module.

    Returns:
        Dictionary containing version, components, and configuration.
    """
    return {
        "version": __version__,
        "author": __author__,
        "components": {
            "long_term_memory": True,
            "working_memory": True,
            "episodic_memory": True,
            "semantic_memory": True,
            "memory_manager": True,
        },
        "exports": sorted(__all__),
        "factory_functions": [
            "create_memory_manager",
            "create_working_memory",
            "create_scratchpad",
            "create_episodic_store",
            "create_semantic_store",
        ],
    }
