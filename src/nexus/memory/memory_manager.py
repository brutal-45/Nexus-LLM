"""
Nexus Memory Manager
====================
Unified memory manager that coordinates all memory subsystems.

The MemoryManager provides a single interface for:
- Processing new information through the full memory pipeline
- Retrieving context from all memory stores simultaneously
- Consolidating short-term memories into long-term storage
- Applying decay and pruning across all stores
- Exporting and importing complete memory state

Architecture:
- Coordinates LongTermMemoryStore, WorkingMemoryBuffer, EpisodicMemoryStore,
  and SemanticMemoryStore
- Routes information to appropriate stores based on content type and metadata
- Merges and ranks retrieval results from multiple stores
- Provides unified statistics and management operations
"""

import os
import json
import time
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from nexus.memory.long_term_memory import (
    MemoryConfig,
    MemoryEntry,
    MemoryEncoder,
    LongTermMemoryStore,
    MemoryConsolidator,
    MemoryDecay,
)
from nexus.memory.working_memory import (
    WorkingMemoryBuffer,
    Scratchpad,
    TaskStateTracker,
)
from nexus.memory.episodic_memory import (
    Episode,
    EpisodicMemoryStore,
)
from nexus.memory.semantic_memory import (
    Fact,
    SemanticMemoryStore,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryManagerConfig:
    """Configuration for the unified MemoryManager.

    Attributes:
        long_term_config: Configuration for the long-term memory store.
        working_memory_capacity: Capacity of the working memory buffer.
        scratchpad_max_length: Maximum character length of the scratchpad.
        enable_episodic: Whether to enable episodic memory.
        enable_semantic: Whether to enable semantic memory.
        episodic_embedding_dim: Embedding dimension for episodic memory.
        semantic_embedding_dim: Embedding dimension for semantic memory.
        decay_rate: Global decay rate for long-term memories.
        auto_consolidate: Whether to automatically consolidate working memories.
        auto_consolidate_interval: Seconds between auto-consolidation runs.
        consolidation_min_importance: Minimum importance for consolidation.
        retrieval_weights: Weight multipliers for each memory store during retrieval.
        max_retrieval_results: Maximum total results across all stores.
        persistence_path: Base path for all memory persistence.
    """
    long_term_config: MemoryConfig = field(default_factory=MemoryConfig)
    working_memory_capacity: int = 64
    scratchpad_max_length: int = 4096
    enable_episodic: bool = True
    enable_semantic: bool = True
    episodic_embedding_dim: int = 256
    semantic_embedding_dim: int = 256
    decay_rate: float = 0.01
    auto_consolidate: bool = False
    auto_consolidate_interval: float = 300.0
    consolidation_min_importance: float = 0.3
    retrieval_weights: Dict[str, float] = field(default_factory=lambda: {
        "long_term": 1.0,
        "episodic": 0.8,
        "semantic": 0.9,
    })
    max_retrieval_results: int = 20
    persistence_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        result["long_term_config"] = self.long_term_config.to_dict()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """A single result from a memory retrieval operation.

    Attributes:
        content: The retrieved text content.
        source: Which memory store this came from ('long_term', 'episodic', 'semantic').
        score: Relevance score after applying source weight.
        raw_score: Original similarity score before weighting.
        metadata: Additional metadata from the source memory.
        memory_id: ID of the memory entry in its source store.
        timestamp: When the memory was created.
        importance: Importance or confidence value.
    """
    content: str = ""
    source: str = ""
    score: float = 0.0
    raw_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_id: str = ""
    timestamp: float = 0.0
    importance: float = 0.5

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "raw_score": self.raw_score,
            "metadata": dict(self.metadata),
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }

    def __repr__(self) -> str:
        preview = self.content[:40] + ("..." if len(self.content) > 40 else "")
        return (
            f"RetrievalResult(source={self.source!r}, score={self.score:.3f}, "
            f"content={preview!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Manager
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """Unified memory manager coordinating all memory subsystems.

    The MemoryManager provides a single entry point for all memory operations,
    automatically routing information to the appropriate stores and merging
    retrieval results from multiple sources.

    Managed Stores:
    - **Long-term memory**: Persistent vector-indexed storage for important information.
    - **Working memory**: Limited-capacity buffer for active reasoning state.
    - **Scratchpad**: Freeform text buffer for chain-of-thought reasoning.
    - **Episodic memory**: Records and recalls past experiences (if enabled).
    - **Semantic memory**: Stores and queries factual knowledge (if enabled).

    Args:
        config: MemoryManagerConfig instance. Uses defaults if None.

    Example:
        >>> config = MemoryManagerConfig(
        ...     persistence_path="./memory_data",
        ...     enable_episodic=True,
        ...     enable_semantic=True,
        ... )
        >>> manager = MemoryManager(config)
        >>> manager.process_input("The user prefers Python", {"source": "preference"})
        >>> results = manager.retrieve_context("What language does the user prefer?")
        >>> for r in results:
        ...     print(f"[{r.source}] {r.content} (score: {r.score:.3f})")
    """

    def __init__(self, config: Optional[MemoryManagerConfig] = None):
        """Initialize the memory manager with all subsystems.

        Args:
            config: Configuration for the manager. Uses defaults if None.
        """
        self.config = config or MemoryManagerConfig()

        # Set up persistence paths
        persistence_base = self.config.persistence_path
        lt_persistence = None
        episodic_persistence = None
        semantic_persistence = None

        if persistence_base:
            os.makedirs(persistence_base, exist_ok=True)
            lt_persistence = os.path.join(persistence_base, "long_term")
            episodic_persistence = os.path.join(persistence_base, "episodic")
            semantic_persistence = os.path.join(persistence_base, "semantic")

        # Configure long-term memory
        lt_config = copy.deepcopy(self.config.long_term_config)
        lt_config.persistence_path = lt_persistence
        lt_config.decay_rate = self.config.decay_rate

        # Initialize all memory stores
        self.long_term = LongTermMemoryStore(config=lt_config)
        self.working_memory = WorkingMemoryBuffer(capacity=self.config.working_memory_capacity)
        self.scratchpad = Scratchpad(max_length=self.config.scratchpad_max_length)
        self.consolidator = MemoryConsolidator(
            memory_store=self.long_term,
            min_importance=self.config.consolidation_min_importance,
        )
        self.decay = MemoryDecay(decay_rate=self.config.decay_rate)

        # Optional stores
        if self.config.enable_episodic:
            self.episodic = EpisodicMemoryStore(
                embedding_dim=self.config.episodic_embedding_dim,
                decay_rate=self.config.decay_rate,
                persistence_path=episodic_persistence,
            )
        else:
            self.episodic = None

        if self.config.enable_semantic:
            self.semantic = SemanticMemoryStore(
                embedding_dim=self.config.semantic_embedding_dim,
                persistence_path=semantic_persistence,
            )
        else:
            self.semantic = None

        # Task state tracker
        self.task_tracker = TaskStateTracker(task_name="memory_manager_task")

        # Timing
        self._last_consolidation: float = time.time()
        self._created_at: float = time.time()

        # Statistics
        self._total_processed: int = 0
        self._total_retrieved: int = 0
        self._total_consolidations: int = 0

    def process_input(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        store_to: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process new information through the full memory pipeline.

        Routes content to appropriate memory stores based on metadata and
        content analysis. Stores to working memory for immediate access,
        and to long-term memory for persistence.

        The routing logic:
        1. Always store in working memory (with metadata key "latest_input")
        2. If metadata contains "category" → also store as semantic fact
        3. If metadata contains "emotional_weight" → also store as episode
        4. If importance >= consolidation threshold → store in long-term memory
        5. Otherwise, store in long-term memory with lower importance

        Args:
            content: The text content to process.
            metadata: Optional metadata dict with keys like:
                - source: Where this content came from
                - category: Semantic category (triggers semantic storage)
                - emotional_weight: Emotional significance (triggers episodic storage)
                - tags: List of tags for categorization
                - type: Explicit store routing ('fact', 'episode', 'memory', 'auto')
            importance: Importance for long-term storage [0.0, 1.0].
            store_to: Explicit list of stores to route to. Options:
                'long_term', 'working', 'episodic', 'semantic', 'scratchpad'.
                If None, uses automatic routing.

        Returns:
            Dictionary with processing results:
            - stored: Dict mapping store names to stored entry IDs
            - skipped: List of stores that were skipped and why
        """
        if not content or not content.strip():
            return {"stored": {}, "skipped": ["all: empty content"]}

        content = content.strip()
        metadata = metadata or {}
        results: Dict[str, str] = {}
        skipped: List[str] = []

        # Determine routing
        if store_to is not None:
            targets = set(store_to)
        else:
            targets = self._auto_route(content, metadata)

        # Store to working memory
        if "working" in targets:
            key = metadata.get("key", f"input_{self._total_processed}")
            self.working_memory.write(key, content)
            results["working"] = key
        elif store_to is None:
            # Always store in working memory by default
            key = f"input_{self._total_processed}"
            self.working_memory.write(key, content)
            results["working"] = key

        # Store to scratchpad
        if "scratchpad" in targets:
            self.scratchpad.append(content)
            results["scratchpad"] = "appended"

        # Store to long-term memory
        if "long_term" in targets:
            entry = self.long_term.store(
                content=content,
                metadata=metadata,
                importance=importance,
            )
            results["long_term"] = entry.id

        # Store to episodic memory
        if "episodic" in targets:
            if self.episodic is not None:
                emotional_weight = metadata.get("emotional_weight", importance)
                tags = metadata.get("tags", [])
                context = metadata.get("context", "")

                episode = self.episodic.record_episode(
                    experience=content,
                    context=context,
                    emotional_weight=emotional_weight,
                    tags=tags,
                    metadata=metadata,
                )
                results["episodic"] = episode.id
            else:
                skipped.append("episodic: disabled")

        # Store to semantic memory
        if "semantic" in targets:
            if self.semantic is not None:
                category = metadata.get("category", "")
                source = metadata.get("source", "")

                fact = self.semantic.store_fact(
                    content=content,
                    source=source,
                    confidence=importance,
                    category=category,
                    metadata=metadata,
                )
                results["semantic"] = fact.id
            else:
                skipped.append("semantic: disabled")

        self._total_processed += 1

        # Auto-consolidate if configured
        if self.config.auto_consolidate:
            self._maybe_auto_consolidate()

        return {"stored": results, "skipped": skipped}

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        sources: Optional[List[str]] = None,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve relevant context from all memory stores.

        Queries all enabled memory stores simultaneously and merges results
        into a unified, ranked list. Results from different stores are
        weighted according to the configuration.

        Args:
            query: The search query text.
            top_k: Maximum total results. If None, uses config.max_retrieval_results.
            sources: List of stores to query. Options: 'long_term', 'episodic', 'semantic'.
                If None, queries all enabled stores.
            category: Optional category filter for semantic memory.
            min_confidence: Minimum confidence/score threshold.

        Returns:
            List of RetrievalResult objects sorted by descending weighted score.
        """
        if not query or not query.strip():
            return []

        if top_k is None:
            top_k = self.config.max_retrieval_results

        all_results: List[RetrievalResult] = []

        # Determine which stores to query
        query_sources = sources if sources is not None else []
        if not query_sources:
            query_sources = ["long_term"]
            if self.episodic is not None:
                query_sources.append("episodic")
            if self.semantic is not None:
                query_sources.append("semantic")

        # Query long-term memory
        if "long_term" in query_sources:
            lt_results = self.long_term.retrieve(query, top_k=top_k * 2)
            weight = self.config.retrieval_weights.get("long_term", 1.0)

            for entry, score in lt_results:
                if score >= min_confidence:
                    result = RetrievalResult(
                        content=entry.content,
                        source="long_term",
                        score=score * weight,
                        raw_score=score,
                        metadata=dict(entry.metadata),
                        memory_id=entry.id,
                        timestamp=entry.created_at,
                        importance=entry.importance,
                    )
                    all_results.append(result)

        # Query episodic memory
        if "episodic" in query_sources and self.episodic is not None:
            ep_results = self.episodic.recall_similar(query, k=top_k * 2, threshold=0.2)
            weight = self.config.retrieval_weights.get("episodic", 0.8)

            for episode, score in ep_results:
                if score >= min_confidence:
                    result = RetrievalResult(
                        content=episode.content,
                        source="episodic",
                        score=score * weight,
                        raw_score=score,
                        metadata={
                            "context": episode.context,
                            "tags": list(episode.tags),
                        },
                        memory_id=episode.id,
                        timestamp=episode.timestamp,
                        importance=episode.emotional_weight,
                    )
                    all_results.append(result)

        # Query semantic memory
        if "semantic" in query_sources and self.semantic is not None:
            sem_results = self.semantic.query_facts(
                query, top_k=top_k * 2,
                category=category,
                min_confidence=min_confidence,
            )
            weight = self.config.retrieval_weights.get("semantic", 0.9)

            for fact, score in sem_results:
                result = RetrievalResult(
                    content=fact.content,
                    source="semantic",
                    score=score * weight * fact.confidence,
                    raw_score=score,
                    metadata={
                        "source": fact.source,
                        "category": fact.category,
                        "confidence": fact.confidence,
                    },
                    memory_id=fact.id,
                    timestamp=fact.created_at,
                    importance=fact.confidence,
                )
                all_results.append(result)

        # Sort by weighted score (descending)
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate similar results
        all_results = self._deduplicate_results(all_results)

        # Apply max results limit
        all_results = all_results[:top_k]

        self._total_retrieved += len(all_results)
        return all_results

    def consolidate(self, min_importance: Optional[float] = None) -> Dict[str, Any]:
        """Consolidate working memories into long-term storage.

        Moves important working memory entries to long-term memory and
        episodic memory (if enabled). Also consolidates episodic memories
        that meet the importance threshold.

        Args:
            min_importance: Override minimum importance for consolidation.

        Returns:
            Dictionary with consolidation results:
            - long_term_consolidated: Number of entries moved to long-term
            - episodic_consolidated: Number of entries moved to episodic
            - lessons_extracted: Number of lessons extracted from episodes
        """
        if min_importance is None:
            min_importance = self.config.consolidation_min_importance

        results = {
            "long_term_consolidated": 0,
            "episodic_consolidated": 0,
            "lessons_extracted": 0,
        }

        # Gather working memory entries
        wm_items = self.working_memory.items()

        if wm_items:
            # Convert working memory entries to MemoryEntry format
            entries = []
            for key, value in wm_items:
                content = str(value) if not isinstance(value, str) else value
                entry = MemoryEntry(
                    id=f"wm_{key}",
                    content=content,
                    metadata={"working_memory_key": key},
                    importance=0.5,  # Default importance for WM entries
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=max(1, self.working_memory.get_access_count(key)),
                )
                entries.append(entry)

            # Consolidate into long-term memory
            consolidated = self.consolidator.consolidate(
                entries,
                target_store=self.long_term,
            )
            results["long_term_consolidated"] = len(consolidated)

            # Also store in episodic memory if enabled
            if self.episodic is not None:
                for entry in entries:
                    if self.consolidator.should_consolidate(entry, min_importance * 0.8):
                        self.episodic.record_episode(
                            experience=entry.content,
                            context="Working memory consolidation",
                            emotional_weight=entry.importance,
                            metadata=entry.metadata,
                        )
                        results["episodic_consolidated"] += 1

        # Extract lessons from episodic memory
        if self.episodic is not None:
            recent_episodes = self.episodic.recall_recent(k=50)
            lessons = self.episodic.extract_lessons(recent_episodes)
            results["lessons_extracted"] = len(lessons)

            # Store lessons as semantic facts
            if self.semantic is not None:
                for lesson_data in lessons:
                    self.semantic.store_fact(
                        content=lesson_data["lesson"],
                        source="lesson_extraction",
                        confidence=lesson_data.get("confidence", 0.5),
                        category="learned_lesson",
                        metadata={
                            "pattern_type": lesson_data.get("pattern_type", ""),
                            "source_episodes": lesson_data.get("source_episodes", []),
                        },
                    )

        self._total_consolidations += 1
        self._last_consolidation = time.time()

        return results

    def forget_unimportant(
        self,
        threshold: float = 0.05,
        apply_decay: bool = True,
    ) -> Dict[str, int]:
        """Apply decay and prune weak memories across all stores.

        Args:
            threshold: Minimum importance/confidence to retain.
            apply_decay: Whether to apply time-based decay before pruning.

        Returns:
            Dictionary with prune counts per store.
        """
        results: Dict[str, int] = {}

        # Long-term memory decay and prune
        if apply_decay:
            all_entries = self.long_term.get_all()
            self.decay.apply_decay(all_entries)

        forgotten = self.long_term.forget(criteria="by_importance", count=50, max_importance=threshold)
        results["long_term_forgotten"] = len(forgotten)

        # Episodic memory decay and prune
        if self.episodic is not None:
            if apply_decay:
                self.episodic.apply_decay()
            pruned = self.episodic.prune_weak(threshold)
            results["episodic_pruned"] = pruned

        # Semantic memory: deprecate low-confidence facts
        if self.semantic is not None:
            all_facts = list(self.semantic._facts.values())
            deprecated = 0
            for fact in all_facts:
                if fact.is_active and fact.confidence < threshold:
                    fact.deprecate("Low confidence during cleanup")
                    deprecated += 1
            results["semantic_deprecated"] = deprecated

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return combined statistics from all memory stores.

        Returns:
            Dictionary with statistics from each store plus overall metrics.
        """
        stats: Dict[str, Any] = {
            "manager": {
                "total_processed": self._total_processed,
                "total_retrieved": self._total_retrieved,
                "total_consolidations": self._total_consolidations,
                "uptime_seconds": time.time() - self._created_at,
                "last_consolidation_seconds_ago": time.time() - self._last_consolidation,
                "stores_enabled": {
                    "long_term": True,
                    "working": True,
                    "scratchpad": True,
                    "episodic": self.episodic is not None,
                    "semantic": self.semantic is not None,
                },
            },
            "long_term": self.long_term.stats(),
            "working_memory": self.working_memory.stats(),
            "scratchpad": self.scratchpad.stats(),
        }

        if self.episodic is not None:
            stats["episodic"] = self.episodic.stats()

        if self.semantic is not None:
            stats["semantic"] = self.semantic.get_statistics()

        stats["decay"] = self.decay.get_stats()
        stats["consolidator"] = self.consolidator.get_stats()

        return stats

    def export_all(self, path: str) -> None:
        """Export all memory state to a directory.

        Creates subdirectories for each memory store and exports their data.

        Args:
            path: Base directory path for exports.
        """
        os.makedirs(path, exist_ok=True)

        # Export long-term memory
        lt_path = os.path.join(path, "long_term_memory.json")
        self.long_term.export_json(lt_path)

        # Export working memory
        wm_path = os.path.join(path, "working_memory.json")
        self.working_memory.export_json(wm_path)

        # Export scratchpad
        sp_path = os.path.join(path, "scratchpad.json")
        self.scratchpad.export_json(sp_path)

        # Export episodic memory
        if self.episodic is not None:
            ep_path = os.path.join(path, "episodic_memory.json")
            self.episodic.export_json(ep_path)

        # Export semantic memory
        if self.semantic is not None:
            sem_path = os.path.join(path, "semantic_memory.json")
            self.semantic.export_json(sem_path)

        # Export manager metadata
        manager_meta = {
            "config": self.config.to_dict(),
            "statistics": {
                "total_processed": self._total_processed,
                "total_retrieved": self._total_retrieved,
                "total_consolidations": self._total_consolidations,
            },
            "exported_at": time.time(),
        }
        meta_path = os.path.join(path, "manager_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(manager_meta, f, indent=2, ensure_ascii=False)

    def import_all(self, path: str, merge: bool = True) -> Dict[str, int]:
        """Import all memory state from a directory.

        Args:
            path: Base directory path containing exported files.
            merge: If True, merges with existing memories. If False, replaces.

        Returns:
            Dictionary with import counts per store.
        """
        results: Dict[str, int] = {}

        # Import long-term memory
        lt_path = os.path.join(path, "long_term_memory.json")
        if os.path.exists(lt_path):
            results["long_term"] = self.long_term.import_json(lt_path, merge=merge)

        # Import working memory
        wm_path = os.path.join(path, "working_memory.json")
        if os.path.exists(wm_path):
            results["working_memory"] = self.working_memory.import_json(wm_path)

        # Import scratchpad
        sp_path = os.path.join(path, "scratchpad.json")
        if os.path.exists(sp_path):
            self.scratchpad.import_json(sp_path)
            results["scratchpad"] = 1

        # Import episodic memory
        if self.episodic is not None:
            ep_path = os.path.join(path, "episodic_memory.json")
            if os.path.exists(ep_path):
                results["episodic"] = self.episodic.import_json(ep_path, merge=merge)

        # Import semantic memory
        if self.semantic is not None:
            sem_path = os.path.join(path, "semantic_memory.json")
            if os.path.exists(sem_path):
                results["semantic"] = self.semantic.import_json(sem_path, merge=merge)

        return results

    def clear_all(self) -> Dict[str, int]:
        """Clear all memory stores.

        Returns:
            Dictionary with counts of cleared items per store.
        """
        results: Dict[str, int] = {}

        results["long_term"] = self.long_term.count()
        self.long_term.clear()

        results["working_memory"] = self.working_memory.clear()
        self.scratchpad.clear()

        if self.episodic is not None:
            results["episodic"] = self.episodic.count()
            # Clear episodic manually since there's no clear method
            for ep_id in list(self.episodic._episodes.keys()):
                self.episodic.delete_episode(ep_id)

        if self.semantic is not None:
            results["semantic"] = self.semantic.clear()

        self.task_tracker.reset()

        return results

    def record_experience(
        self,
        experience: str,
        context: str = "",
        emotional_weight: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> Optional[Episode]:
        """Convenience method to record an episodic memory.

        Args:
            experience: Description of the experience.
            context: Situational context.
            emotional_weight: Emotional significance [0.0, 1.0].
            tags: Categorization tags.

        Returns:
            Created Episode, or None if episodic memory is disabled.
        """
        if self.episodic is None:
            return None

        return self.episodic.record_episode(
            experience=experience,
            context=context,
            emotional_weight=emotional_weight,
            tags=tags,
        )

    def store_fact(
        self,
        content: str,
        source: str = "",
        confidence: float = 0.5,
        category: str = "",
    ) -> Optional[Fact]:
        """Convenience method to store a semantic fact.

        Args:
            content: The factual statement.
            source: Provenance.
            confidence: Confidence score [0.0, 1.0].
            category: Category label.

        Returns:
            Created Fact, or None if semantic memory is disabled.
        """
        if self.semantic is None:
            return None

        return self.semantic.store_fact(
            content=content,
            source=source,
            confidence=confidence,
            category=category,
        )

    def remember(self, content: str, importance: float = 0.5, metadata: Optional[Dict] = None) -> MemoryEntry:
        """Convenience method to store content in long-term memory.

        Args:
            content: Content to remember.
            importance: Importance score [0.0, 1.0].
            metadata: Optional metadata.

        Returns:
            Created MemoryEntry.
        """
        return self.long_term.store(
            content=content,
            metadata=metadata or {},
            importance=importance,
        )

    def recall(self, query: str, top_k: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Convenience method to retrieve from long-term memory.

        Args:
            query: Search query.
            top_k: Maximum results.

        Returns:
            List of (MemoryEntry, score) tuples.
        """
        return self.long_term.retrieve(query, top_k=top_k)

    def get_working_memory_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current working memory state.

        Returns:
            Working memory snapshot dictionary.
        """
        return self.working_memory.snapshot()

    def get_scratchpad_content(self) -> str:
        """Get the current scratchpad content.

        Returns:
            Scratchpad text content.
        """
        return self.scratchpad.read()

    def get_task_progress(self) -> float:
        """Get the current task progress.

        Returns:
            Progress fraction [0.0, 1.0].
        """
        return self.task_tracker.progress()

    def _auto_route(self, content: str, metadata: Dict[str, Any]) -> Set[str]:
        """Automatically determine which stores to route content to.

        Args:
            content: The content text.
            metadata: Content metadata.

        Returns:
            Set of store names to route to.
        """
        targets = {"long_term", "working"}

        # Check for explicit type
        content_type = metadata.get("type", "auto")

        if content_type == "fact":
            targets.add("semantic")
        elif content_type == "episode":
            targets.add("episodic")
        elif content_type == "reasoning":
            targets.add("scratchpad")
        elif content_type == "auto":
            # Auto-detect based on metadata
            if metadata.get("category"):
                targets.add("semantic")
            if metadata.get("emotional_weight"):
                targets.add("episodic")
            if metadata.get("context"):
                targets.add("episodic")

            # Analyze content for fact-like patterns
            if self._looks_like_fact(content):
                targets.add("semantic")

        return targets

    def _looks_like_fact(self, text: str) -> bool:
        """Heuristic check if text looks like a factual statement.

        Checks for patterns common in factual statements:
        - Contains dates, numbers, proper nouns
        - Uses declarative language
        - Contains "is", "was", "are", "were"
        - Contains specific entities

        Args:
            text: Input text.

        Returns:
            True if text appears to be a factual statement.
        """
        import re

        # Check for date patterns
        date_patterns = [
            r'\b\d{4}\b',  # Year
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        ]
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Check for copula verbs (common in factual statements)
        copula_pattern = r'\b(?:is|are|was|were|has|have|had)\b'
        if re.search(copula_pattern, text, re.IGNORECASE):
            # Also check it's not a question
            if not text.strip().endswith("?"):
                word_count = len(text.split())
                if word_count >= 5:
                    return True

        # Check for specific entities (capitalized words not at start)
        words = text.split()
        capitalized = [w for w in words[1:] if w[0:1].isupper() and w[0:1].isalpha()]
        if len(capitalized) >= 2:
            return True

        return False

    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove near-duplicate retrieval results.

        Keeps the highest-scoring result when duplicates are found.

        Args:
            results: List of retrieval results.

        Returns:
            Deduplicated list.
        """
        if not results:
            return []

        unique: List[RetrievalResult] = []
        seen_content: Dict[str, float] = {}

        for result in results:
            # Normalize content for comparison
            normalized = result.content.lower().strip()[:80]

            # Check for near-duplicates
            is_dup = False
            for seen_norm, seen_score in seen_content.items():
                words_seen = set(seen_norm.split())
                words_new = set(normalized.split())
                if words_seen and words_new:
                    overlap = len(words_seen & words_new) / len(words_seen | words_new)
                    if overlap > 0.7:
                        is_dup = True
                        break

            if not is_dup:
                unique.append(result)
                seen_content[normalized] = result.score

        return unique

    def _maybe_auto_consolidate(self) -> None:
        """Run auto-consolidation if enough time has passed."""
        if not self.config.auto_consolidate:
            return

        now = time.time()
        if now - self._last_consolidation >= self.config.auto_consolidate_interval:
            self.consolidate()

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"long_term={len(self.long_term)}, "
            f"working={self.working_memory.size()}/{self.working_memory.capacity()}, "
            f"episodic={len(self.episodic) if self.episodic else 'disabled'}, "
            f"semantic={self.semantic.count() if self.semantic else 'disabled'})"
        )
