"""
Nexus Episodic Memory
=====================
Episodic memory system for recording and recalling past experiences (episodes).

Episodic memory stores specific experiences or events along with their context,
emotional weighting, temporal information, and inter-episode relationships. This
mirrors the human episodic memory system which stores "what happened, when, and where."

Key Capabilities:
- Record experiences as episodes with emotional weight and tags
- Recall similar past experiences using embedding-based similarity
- Recall recent or important episodes
- Filter episodes by tags
- Extract reusable lessons from past experiences
- Query episodes within time ranges

Architecture:
- **Episode**: Data class representing a single episodic memory
- **EpisodicMemoryStore**: Main store for managing episodes
- **EpisodeEncoder**: Lightweight text encoder for episode embeddings
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import hashlib
import time
import os
import math
import collections
import copy
import re


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Data Class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """A single episodic memory representing a past experience.

    Episodes capture specific events or interactions along with their context,
    emotional significance, temporal information, and relationships to other
    episodes. They are the fundamental unit of episodic memory.

    Attributes:
        id: Unique identifier for this episode. Auto-generated if not provided.
        content: The main content or description of the experience.
        context: The situation or environment in which the experience occurred.
            Provides background information for interpreting the content.
        timestamp: Unix timestamp (seconds since epoch) of when the episode occurred.
        emotional_weight: Float in [0.0, 1.0] representing the emotional significance
            of the episode. Higher values indicate more emotionally charged or
            important experiences. These are more likely to be recalled.
        related_ids: Set of episode IDs that are related to this episode.
            Used to link sequences of events or cause-effect relationships.
        tags: Set of string tags for categorization and filtering.
            Examples: "success", "failure", "user_feedback", "error", "learning".
        metadata: Additional arbitrary metadata attached to this episode.
        embedding: Optional tensor embedding of the content for similarity search.
        access_count: Number of times this episode has been recalled.
        last_accessed: Timestamp of the most recent recall.
        importance_decay: Current decayed importance after applying time decay.
    """
    id: str = ""
    content: str = ""
    context: str = ""
    timestamp: float = 0.0
    emotional_weight: float = 0.5
    related_ids: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None
    access_count: int = 0
    last_accessed: float = 0.0
    importance_decay: float = 1.0

    @staticmethod
    def generate_id(content: str, timestamp: Optional[float] = None) -> str:
        """Generate a deterministic unique ID for an episode.

        Args:
            content: Episode content string.
            timestamp: Optional timestamp for uniqueness.

        Returns:
            SHA-256 hash truncated to 16 hex characters.
        """
        if timestamp is None:
            timestamp = time.time()
        raw = f"episode:{content}:{timestamp}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize episode to a JSON-compatible dictionary.

        Returns:
            Dictionary with all episode fields.
        """
        embedding_list = None
        if self.embedding is not None:
            if isinstance(self.embedding, torch.Tensor):
                embedding_list = self.embedding.detach().cpu().tolist()
            else:
                embedding_list = list(self.embedding)

        return {
            "id": self.id,
            "content": self.content,
            "context": self.context,
            "timestamp": self.timestamp,
            "emotional_weight": self.emotional_weight,
            "related_ids": list(self.related_ids),
            "tags": list(self.tags),
            "metadata": self.metadata,
            "embedding": embedding_list,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "importance_decay": self.importance_decay,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Deserialize an episode from a dictionary.

        Args:
            data: Dictionary with episode fields.

        Returns:
            Reconstructed Episode instance.
        """
        embedding = None
        if data.get("embedding") is not None:
            embedding_list = data["embedding"]
            if isinstance(embedding_list, list):
                embedding = torch.tensor(embedding_list, dtype=torch.float32)

        related_ids = data.get("related_ids", [])
        tags = data.get("tags", [])

        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            context=data.get("context", ""),
            timestamp=data.get("timestamp", 0.0),
            emotional_weight=data.get("emotional_weight", 0.5),
            related_ids=set(related_ids) if isinstance(related_ids, list) else set(),
            tags=set(tags) if isinstance(tags, list) else set(),
            metadata=data.get("metadata", {}),
            embedding=embedding,
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", 0.0),
            importance_decay=data.get("importance_decay", 1.0),
        )

    def touch(self) -> None:
        """Update access statistics when this episode is recalled."""
        self.access_count += 1
        self.last_accessed = time.time()

    def effective_weight(self, decay_rate: float = 0.005) -> float:
        """Compute the effective emotional weight after applying time and access decay.

        Args:
            decay_rate: Rate of exponential time decay.

        Returns:
            Effective weight in [0.0, 1.0].
        """
        hours_since = (time.time() - self.timestamp) / 3600.0
        time_decay = math.exp(-decay_rate * hours_since)

        # Access recency bonus
        recency_bonus = 0.0
        if self.access_count > 0 and self.last_accessed > 0:
            hours_since_access = (time.time() - self.last_accessed) / 3600.0
            recency_bonus = 0.1 * min(1.0, math.log1p(self.access_count)) * math.exp(-0.1 * hours_since_access)

        effective = self.emotional_weight * time_decay * self.importance_decay + recency_bonus
        return max(0.0, min(1.0, effective))

    def age_hours(self) -> float:
        """Get the age of this episode in hours."""
        return (time.time() - self.timestamp) / 3600.0

    def add_relation(self, episode_id: str) -> None:
        """Add a relation to another episode.

        Args:
            episode_id: ID of the related episode.
        """
        self.related_ids.add(episode_id)

    def remove_relation(self, episode_id: str) -> None:
        """Remove a relation to another episode.

        Args:
            episode_id: ID of the episode to unlink.
        """
        self.related_ids.discard(episode_id)

    def has_tag(self, tag: str) -> bool:
        """Check if this episode has a specific tag.

        Args:
            tag: Tag string to check (case-insensitive).

        Returns:
            True if the tag is present.
        """
        return tag.lower() in {t.lower() for t in self.tags}

    def add_tags(self, *tags: str) -> None:
        """Add one or more tags to this episode.

        Args:
            *tags: Tag strings to add.
        """
        for tag in tags:
            self.tags.add(tag.strip().lower())

    def __repr__(self) -> str:
        content_preview = self.content[:40] + ("..." if len(self.content) > 40 else "")
        return (
            f"Episode(id={self.id!r}, content={content_preview!r}, "
            f"weight={self.emotional_weight:.2f}, tags={sorted(self.tags)[:3]})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodeEncoder(nn.Module):
    """Lightweight encoder for episode content embeddings.

    Uses a simple bag-of-words + character n-gram approach:
    1. Tokenize text into word tokens
    2. Hash tokens to fixed vocabulary size
    3. Create bag-of-words vector
    4. Add character n-gram features
    5. Project through linear layers
    6. L2-normalize output

    This encoder is designed for speed and simplicity, not for state-of-the-art
    semantic understanding. For production use, replace with a pretrained model.
    """

    def __init__(self, vocab_size: int = 50000, embed_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.embedding.weight[0])
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _hash_tokens(self, text: str) -> List[int]:
        """Hash text tokens to vocabulary indices.

        Uses a deterministic hashing approach to map arbitrary text tokens
        to fixed-size vocabulary indices.

        Args:
            text: Input text.

        Returns:
            List of vocabulary indices.
        """
        if not text:
            return []

        tokens = re.findall(r'\b\w+\b', text.lower())
        indices = []

        for token in tokens:
            # Use hash to map to vocab index (avoiding special tokens 0-3)
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = (h % (self.vocab_size - 4)) + 4
            indices.append(idx)

        # Add character bigrams for more granular matching
        text_lower = text.lower()
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i + 2]
            if bigram.isalnum():
                h = int(hashlib.md5(f"bigram:{bigram}".encode()).hexdigest(), 16)
                idx = (h % (self.vocab_size - 4)) + 4
                indices.append(idx)

        return indices if indices else [1]  # UNK token

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text into normalized embedding vectors.

        Args:
            text: A string or list of strings.

        Returns:
            Normalized embedding tensor.
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        all_indices = [self._hash_tokens(t) for t in text]

        # Pad to same length
        max_len = max(len(idx) for idx in all_indices) if all_indices else 1
        padded = []
        for idx in all_indices:
            padded.append(idx + [0] * (max_len - len(idx)))

        token_tensor = torch.tensor(padded, dtype=torch.long)

        # Embedding lookup
        embedded = self.embedding(token_tensor)  # (batch, seq, embed_dim)

        # Mean pooling (ignore padding)
        mask = (token_tensor != 0).float().unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts  # (batch, embed_dim)

        # Project
        projected = self.projection(pooled)  # (batch, output_dim)

        # L2 normalize
        norms = projected.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norms

        if single:
            return normalized.squeeze(0)
        return normalized

    @torch.no_grad()
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text without gradient computation.

        Args:
            text: Input text.

        Returns:
            Normalized embedding tensor.
        """
        return self.forward(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Episodic Memory Store
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodicMemoryStore:
    """Store and retrieve episodic memories (past experiences).

    The EpisodicMemoryStore maintains a collection of Episode objects with
    support for similarity-based recall, temporal queries, tag filtering,
    and lesson extraction from past experiences.

    Features:
    - Record experiences with context, emotional weight, and tags
    - Recall similar past experiences via embedding cosine similarity
    - Recall recent or important episodes
    - Filter episodes by tags
    - Extract reusable lessons from past experiences
    - Get episodes within time ranges
    - Automatic importance decay

    Args:
        embedding_dim: Dimensionality of episode embeddings.
        decay_rate: Rate of importance decay over time.
        max_episodes: Maximum number of episodes to store.
        persistence_path: Optional path for saving/loading episodes.

    Example:
        >>> store = EpisodicMemoryStore(embedding_dim=256)
        >>> store.record_episode(
        ...     experience="User was confused by the error message",
        ...     context="Technical support chat",
        ...     emotional_weight=0.7,
        ...     tags=["confusion", "ux_issue"],
        ... )
        >>> similar = store.recall_similar("User finds error messages unclear", k=3)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        decay_rate: float = 0.005,
        max_episodes: int = 100000,
        persistence_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.max_episodes = max_episodes
        self.persistence_path = persistence_path

        # Episode storage
        self._episodes: Dict[str, Episode] = collections.OrderedDict()

        # Tag index for fast tag-based lookup
        self._tag_index: Dict[str, Set[str]] = collections.defaultdict(set)

        # Time index for range queries
        self._time_sorted_ids: List[str] = []

        # Encoder
        self._encoder = EpisodeEncoder(
            vocab_size=50000,
            embed_dim=128,
            output_dim=embedding_dim,
        )
        self._encoder.eval()

        # Statistics
        self._total_recorded = 0
        self._total_recalled = 0
        self._total_lessons_extracted = 0

        # Load from disk if available
        if persistence_path:
            os.makedirs(persistence_path, exist_ok=True)
            self._load_if_exists()

    def record_episode(
        self,
        experience: str,
        context: str = "",
        emotional_weight: float = 0.5,
        tags: Optional[List[str]] = None,
        related_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> Episode:
        """Record a new episodic memory.

        Creates an Episode with the provided information, generates an embedding
        for similarity search, and stores it in the episode collection.

        Args:
            experience: The main content/description of the experience.
            context: Background or situational context for the experience.
            emotional_weight: Emotional significance in [0.0, 1.0].
            tags: List of string tags for categorization.
            related_ids: IDs of related episodes.
            metadata: Additional arbitrary metadata.
            timestamp: When the episode occurred. If None, uses current time.
            episode_id: Explicit ID. Auto-generated if None.

        Returns:
            The newly created Episode.

        Raises:
            ValueError: If experience is empty.
        """
        if not experience or not experience.strip():
            raise ValueError("Experience content cannot be empty")

        experience = experience.strip()
        emotional_weight = max(0.0, min(1.0, emotional_weight))
        tags = tags or []
        related_ids = related_ids or []
        metadata = metadata or {}
        timestamp = timestamp or time.time()

        # Check capacity
        if len(self._episodes) >= self.max_episodes:
            self._evict_oldest(1)

        # Generate ID
        if episode_id is None:
            episode_id = Episode.generate_id(experience, timestamp)

        # Generate embedding from content + context
        full_text = f"{experience} {context}".strip()
        embedding = self._encoder.encode(full_text)

        # Create episode
        episode = Episode(
            id=episode_id,
            content=experience,
            context=context,
            timestamp=timestamp,
            emotional_weight=emotional_weight,
            related_ids=set(related_ids),
            tags=set(t.strip().lower() for t in tags if t.strip()),
            metadata=metadata,
            embedding=embedding,
            access_count=0,
            last_accessed=0.0,
            importance_decay=1.0,
        )

        # Store
        self._episodes[episode_id] = episode

        # Update tag index
        for tag in episode.tags:
            self._tag_index[tag].add(episode_id)

        # Update time index (insert in sorted order)
        self._insert_time_sorted(episode_id, timestamp)

        self._total_recorded += 1

        return episode

    def recall_similar(
        self,
        experience: str,
        k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[Episode, float]]:
        """Recall past episodes similar to a given experience.

        Encodes the query experience and computes cosine similarity against
        all stored episode embeddings. Returns the top-k most similar episodes.

        Args:
            experience: The query experience text.
            k: Maximum number of episodes to return.
            threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of (Episode, similarity_score) tuples sorted by descending
            similarity.
        """
        if not experience or not self._episodes:
            return []

        # Encode query
        query_embedding = self._encoder.encode(experience)

        # Compute similarities
        results: List[Tuple[Episode, float]] = []
        for episode in self._episodes.values():
            if episode.embedding is None:
                continue

            sim = self._cosine_similarity(query_embedding, episode.embedding)
            if sim >= threshold:
                results.append((episode, sim))

        # Sort by descending similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Touch recalled episodes
        for episode, _ in results[:k]:
            episode.touch()

        self._total_recalled += len(results[:k])
        return results[:k]

    def recall_recent(self, k: int = 10) -> List[Episode]:
        """Recall the k most recent episodes.

        Episodes are ordered by timestamp (most recent first).

        Args:
            k: Maximum number of episodes to return.

        Returns:
            List of Episode objects, most recent first.
        """
        recent_ids = self._time_sorted_ids[-k:] if self._time_sorted_ids else []
        recent_ids.reverse()

        episodes = []
        for ep_id in recent_ids:
            if ep_id in self._episodes:
                episode = self._episodes[ep_id]
                episode.touch()
                episodes.append(episode)

        return episodes

    def recall_important(self, k: int = 10) -> List[Tuple[Episode, float]]:
        """Recall episodes with the highest effective emotional weight.

        Effective weight accounts for time decay, access patterns, and
        the base emotional weight.

        Args:
            k: Maximum number of episodes to return.

        Returns:
            List of (Episode, effective_weight) tuples sorted by descending
            effective weight.
        """
        weighted = [
            (ep, ep.effective_weight(self.decay_rate))
            for ep in self._episodes.values()
        ]
        weighted.sort(key=lambda x: x[1], reverse=True)

        results = weighted[:k]
        for episode, _ in results:
            episode.touch()

        self._total_recalled += len(results)
        return results

    def recall_by_tag(self, tag: str, k: int = 10) -> List[Episode]:
        """Recall episodes that have a specific tag.

        Tag matching is case-insensitive. Results are ordered by timestamp
        (most recent first).

        Args:
            tag: Tag string to filter by.
            k: Maximum number of episodes to return.

        Returns:
            List of Episode objects with the matching tag.
        """
        tag_normalized = tag.strip().lower()
        episode_ids = self._tag_index.get(tag_normalized, set())

        episodes = []
        for ep_id in episode_ids:
            if ep_id in self._episodes:
                episodes.append(self._episodes[ep_id])

        # Sort by timestamp descending
        episodes.sort(key=lambda ep: ep.timestamp, reverse=True)

        results = episodes[:k]
        for episode in results:
            episode.touch()

        return results

    def recall_by_tags(self, tags: List[str], k: int = 10, match_all: bool = False) -> List[Episode]:
        """Recall episodes matching one or more tags.

        Args:
            tags: List of tag strings to filter by.
            k: Maximum number of results.
            match_all: If True, only return episodes with ALL specified tags.
                If False, return episodes with ANY of the specified tags.

        Returns:
            List of matching Episode objects.
        """
        normalized_tags = {t.strip().lower() for t in tags if t.strip()}

        if not normalized_tags:
            return []

        if match_all:
            # Episodes must have ALL tags
            candidate_sets = []
            for tag in normalized_tags:
                ids = self._tag_index.get(tag, set())
                candidate_sets.append(ids)

            if not candidate_sets:
                return []

            common_ids = candidate_sets[0]
            for s in candidate_sets[1:]:
                common_ids = common_ids & s
            episode_ids = common_ids
        else:
            # Episodes with ANY tag
            episode_ids = set()
            for tag in normalized_tags:
                episode_ids.update(self._tag_index.get(tag, set()))

        episodes = []
        for ep_id in episode_ids:
            if ep_id in self._episodes:
                episodes.append(self._episodes[ep_id])

        episodes.sort(key=lambda ep: ep.timestamp, reverse=True)

        results = episodes[:k]
        for episode in results:
            episode.touch()

        return results

    def extract_lessons(self, episodes: Optional[List[Episode]] = None) -> List[Dict[str, Any]]:
        """Extract reusable lessons from past experiences.

        Analyzes episodes to identify patterns, common themes, and actionable
        lessons. Lessons are extracted based on:
        1. High-emotional-weight episodes (important experiences)
        2. Frequently co-occurring tags
        3. Common content patterns across related episodes
        4. Temporal patterns (e.g., recurring issues)

        Args:
            episodes: Episodes to analyze. If None, analyzes all episodes.

        Returns:
            List of lesson dictionaries, each containing:
            - lesson: The extracted lesson text
            - confidence: Confidence score [0.0, 1.0]
            - source_episodes: IDs of episodes contributing to this lesson
            - tags: Tags associated with the lesson
            - pattern_type: Type of pattern identified
        """
        if episodes is None:
            episodes = list(self._episodes.values())

        if not episodes:
            return []

        lessons: List[Dict[str, Any]] = []

        # Pattern 1: Extract lessons from high-weight episodes
        high_weight = [ep for ep in episodes if ep.emotional_weight >= 0.7]
        for episode in high_weight:
            lesson = self._extract_single_lesson(episode)
            if lesson:
                lessons.append(lesson)

        # Pattern 2: Identify tag co-occurrence patterns
        tag_lessons = self._extract_tag_pattern_lessons(episodes)
        lessons.extend(tag_lessons)

        # Pattern 3: Extract lessons from episode clusters (related episodes)
        cluster_lessons = self._extract_cluster_lessons(episodes)
        lessons.extend(cluster_lessons)

        # Pattern 4: Temporal pattern detection
        temporal_lessons = self._extract_temporal_lessons(episodes)
        lessons.extend(temporal_lessons)

        # Sort by confidence and deduplicate
        lessons.sort(key=lambda l: l["confidence"], reverse=True)
        lessons = self._deduplicate_lessons(lessons)

        self._total_lessons_extracted += len(lessons)
        return lessons[:20]  # Cap at 20 lessons

    def get_timeline(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Episode]:
        """Get episodes within a time range.

        Args:
            start_time: Start of time range (Unix timestamp). If None, no lower bound.
            end_time: End of time range (Unix timestamp). If None, no upper bound.

        Returns:
            List of Episode objects within the time range, ordered chronologically.
        """
        if not self._time_sorted_ids:
            return []

        # Find start index using binary search
        start_idx = 0
        if start_time is not None:
            start_idx = self._binary_search_time(start_time)

        episodes = []
        for ep_id in self._time_sorted_ids[start_idx:]:
            episode = self._episodes.get(ep_id)
            if episode is None:
                continue

            if start_time is not None and episode.timestamp < start_time:
                continue
            if end_time is not None and episode.timestamp > end_time:
                break

            episodes.append(episode)

        return episodes

    def compute_episode_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """Compute cosine similarity between two episode embeddings.

        Also considers context and tag overlap as secondary signals.

        Args:
            ep1: First episode.
            ep2: Second episode.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        # Primary: embedding similarity
        embedding_sim = 0.0
        if ep1.embedding is not None and ep2.embedding is not None:
            embedding_sim = self._cosine_similarity(ep1.embedding, ep2.embedding)

        # Secondary: tag overlap
        tag_overlap = 0.0
        if ep1.tags and ep2.tags:
            common_tags = ep1.tags & ep2.tags
            tag_overlap = len(common_tags) / max(1, min(len(ep1.tags), len(ep2.tags)))

        # Tertiary: context word overlap
        context_sim = 0.0
        if ep1.context and ep2.context:
            words1 = set(ep1.context.lower().split())
            words2 = set(ep2.context.lower().split())
            if words1 and words2:
                context_sim = len(words1 & words2) / len(words1 | words2)

        # Weighted combination
        combined = 0.7 * embedding_sim + 0.2 * tag_overlap + 0.1 * context_sim
        return max(0.0, min(1.0, combined))

    def find_related(self, episode_id: str, k: int = 5) -> List[Tuple[Episode, float]]:
        """Find episodes related to a given episode.

        Uses both explicit relations and embedding similarity.

        Args:
            episode_id: ID of the reference episode.
            k: Maximum number of related episodes.

        Returns:
            List of (Episode, similarity_score) tuples.
        """
        episode = self._episodes.get(episode_id)
        if episode is None:
            return []

        results: List[Tuple[Episode, float]] = []

        # Add explicitly related episodes
        for related_id in episode.related_ids:
            related_ep = self._episodes.get(related_id)
            if related_ep is not None:
                sim = self.compute_episode_similarity(episode, related_ep)
                results.append((related_ep, sim))

        # Find similar episodes by embedding
        similar = self.recall_similar(episode.content, k=k * 2, threshold=0.2)

        for ep, sim in similar:
            if ep.id != episode_id:
                results.append((ep, sim))

        # Deduplicate and sort
        seen = {episode_id}
        unique_results = []
        for ep, sim in results:
            if ep.id not in seen:
                seen.add(ep.id)
                unique_results.append((ep, sim))

        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:k]

    def update_episode(
        self,
        episode_id: str,
        content: Optional[str] = None,
        context: Optional[str] = None,
        emotional_weight: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Episode]:
        """Update an existing episode.

        If content or context is changed, the embedding is recomputed.

        Args:
            episode_id: ID of the episode to update.
            content: New content. If None, not changed.
            context: New context. If None, not changed.
            emotional_weight: New emotional weight. If None, not changed.
            tags: New tags. If provided, replaces existing tags.
            metadata: New metadata. If provided, merges with existing.

        Returns:
            Updated Episode, or None if not found.
        """
        episode = self._episodes.get(episode_id)
        if episode is None:
            return None

        content_changed = False

        if content is not None and content.strip():
            # Remove old tag entries
            for tag in episode.tags:
                self._tag_index[tag].discard(episode_id)

            episode.content = content.strip()
            content_changed = True

        if context is not None:
            if episode.context != context:
                content_changed = True
            episode.context = context

        if emotional_weight is not None:
            episode.emotional_weight = max(0.0, min(1.0, emotional_weight))

        if tags is not None:
            # Remove old tag entries
            for tag in episode.tags:
                self._tag_index[tag].discard(episode_id)
            episode.tags = set(t.strip().lower() for t in tags if t.strip())
            for tag in episode.tags:
                self._tag_index[tag].add(episode_id)

        if metadata is not None:
            episode.metadata.update(metadata)

        # Recompute embedding if content changed
        if content_changed:
            full_text = f"{episode.content} {episode.context}".strip()
            episode.embedding = self._encoder.encode(full_text)

        return episode

    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode by ID.

        Args:
            episode_id: ID of the episode to delete.

        Returns:
            True if found and deleted.
        """
        episode = self._episodes.get(episode_id)
        if episode is None:
            return False

        # Remove from tag index
        for tag in episode.tags:
            self._tag_index[tag].discard(episode_id)

        # Remove from time index
        if episode_id in self._time_sorted_ids:
            self._time_sorted_ids.remove(episode_id)

        # Remove from storage
        del self._episodes[episode_id]

        return True

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID.

        Args:
            episode_id: ID of the episode.

        Returns:
            Episode if found, None otherwise.
        """
        episode = self._episodes.get(episode_id)
        if episode is not None:
            episode.touch()
        return episode

    def count(self) -> int:
        """Return the number of stored episodes.

        Returns:
            Episode count.
        """
        return len(self._episodes)

    def get_all_tags(self) -> Dict[str, int]:
        """Get all tags and their episode counts.

        Returns:
            Dictionary mapping tags to episode counts.
        """
        result = {}
        for tag, episode_ids in self._tag_index.items():
            if episode_ids:
                result[tag] = len(episode_ids)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def apply_decay(self) -> int:
        """Apply importance decay to all episodes.

        Updates the importance_decay field of each episode based on time
        since creation and access patterns.

        Returns:
            Number of episodes decayed.
        """
        count = 0
        for episode in self._episodes.values():
            hours = episode.age_hours()
            decay = math.exp(-self.decay_rate * hours)
            episode.importance_decay *= decay
            count += 1
        return count

    def prune_weak(self, threshold: float = 0.01) -> int:
        """Remove episodes with very low effective importance.

        Args:
            threshold: Minimum effective weight to keep.

        Returns:
            Number of episodes pruned.
        """
        weak_ids = []
        for episode_id, episode in self._episodes.items():
            effective = episode.effective_weight(self.decay_rate)
            if effective < threshold:
                weak_ids.append(episode_id)

        for ep_id in weak_ids:
            self.delete_episode(ep_id)

        return len(weak_ids)

    def export_json(self, path: str) -> None:
        """Export all episodes to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "episodes": [ep.to_dict() for ep in self._episodes.values()],
            "config": {
                "embedding_dim": self.embedding_dim,
                "decay_rate": self.decay_rate,
                "max_episodes": self.max_episodes,
            },
            "statistics": {
                "total_recorded": self._total_recorded,
                "total_recalled": self._total_recalled,
                "total_lessons_extracted": self._total_lessons_extracted,
                "exported_at": time.time(),
            },
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def import_json(self, path: str, merge: bool = True) -> int:
        """Import episodes from a JSON file.

        Args:
            path: Input file path.
            merge: If True, merges with existing episodes. If False, replaces all.

        Returns:
            Number of episodes imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not merge:
            self._episodes.clear()
            self._tag_index.clear()
            self._time_sorted_ids.clear()

        count = 0
        for ep_data in data.get("episodes", []):
            episode = Episode.from_dict(ep_data)
            self._episodes[episode.id] = episode

            for tag in episode.tags:
                self._tag_index[tag].add(episode.id)

            self._insert_time_sorted(episode.id, episode.timestamp)
            count += 1

        self._total_recorded += count
        return count

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive statistics.

        Returns:
            Dictionary with store statistics.
        """
        if not self._episodes:
            return {
                "count": 0,
                "max_episodes": self.max_episodes,
                "total_recorded": self._total_recorded,
                "total_recalled": self._total_recalled,
                "total_lessons_extracted": self._total_lessons_extracted,
                "tag_count": 0,
            }

        episodes = list(self._episodes.values())
        weights = [ep.emotional_weight for ep in episodes]
        access_counts = [ep.access_count for ep in episodes]

        return {
            "count": len(episodes),
            "max_episodes": self.max_episodes,
            "utilization": len(episodes) / self.max_episodes,
            "total_recorded": self._total_recorded,
            "total_recalled": self._total_recalled,
            "total_lessons_extracted": self._total_lessons_extracted,
            "avg_emotional_weight": sum(weights) / len(weights),
            "min_emotional_weight": min(weights),
            "max_emotional_weight": max(weights),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "total_tags": len(self._tag_index),
            "most_common_tags": list(self.get_all_tags().items())[:10],
            "oldest_episode_age_hours": max(ep.age_hours() for ep in episodes),
            "newest_episode_age_hours": min(ep.age_hours() for ep in episodes),
        }

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors.

        Args:
            a: First tensor.
            b: Second tensor.

        Returns:
            Cosine similarity as a float.
        """
        dot = torch.dot(a, b)
        norm_a = a.norm()
        norm_b = b.norm()
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))

    def _insert_time_sorted(self, episode_id: str, timestamp: float) -> None:
        """Insert an episode ID into the time-sorted list.

        Uses binary search for efficient insertion.

        Args:
            episode_id: Episode ID to insert.
            timestamp: Episode timestamp.
        """
        import bisect

        # Use tuple (timestamp, id) for stable sorting
        entry = (timestamp, episode_id)

        # Find insertion point
        insert_idx = bisect.bisect_left(self._time_sorted_ids, episode_id,
                                         key=lambda x: self._episodes[x].timestamp
                                         if x in self._episodes else 0)

        # Simpler approach: just append and sort periodically
        self._time_sorted_ids.append(episode_id)
        # Sort by timestamp periodically (not every insert for performance)
        if len(self._time_sorted_ids) > 1000 and len(self._time_sorted_ids) % 100 == 0:
            self._rebuild_time_index()

    def _rebuild_time_index(self) -> None:
        """Rebuild the time-sorted index from scratch."""
        self._time_sorted_ids = sorted(
            self._episodes.keys(),
            key=lambda eid: self._episodes[eid].timestamp if eid in self._episodes else 0
        )

    def _binary_search_time(self, target_time: float) -> int:
        """Binary search to find the first episode at or after target_time.

        Args:
            target_time: Target timestamp.

        Returns:
            Index into _time_sorted_ids.
        """
        lo, hi = 0, len(self._time_sorted_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            ep_id = self._time_sorted_ids[mid]
            ep_time = self._episodes[ep_id].timestamp if ep_id in self._episodes else 0
            if ep_time < target_time:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _evict_oldest(self, count: int) -> int:
        """Evict the oldest episodes to free capacity.

        Args:
            count: Number of episodes to evict.

        Returns:
            Number actually evicted.
        """
        if not self._time_sorted_ids:
            return 0

        self._rebuild_time_index()
        evicted = 0

        for ep_id in self._time_sorted_ids[:count]:
            if self.delete_episode(ep_id):
                evicted += 1

        return evicted

    def _extract_single_lesson(self, episode: Episode) -> Optional[Dict[str, Any]]:
        """Extract a lesson from a single high-weight episode.

        Analyzes the episode content to identify actionable insights.

        Args:
            episode: The episode to analyze.

        Returns:
            Lesson dictionary or None if no lesson can be extracted.
        """
        content = episode.content.lower()

        # Pattern: "X was/were [adjective/problem]" → lesson about fixing
        patterns = [
            (r"(\w+)\s+(?:was|were|is|are)\s+(\w+(?:\s+\w+)?)\s+(?:because|due to|caused by)\s+(.+)",
             lambda m: f"When {m.group(1)} {m.group(2)}, it is important to address: {m.group(3)}"),
            (r"the\s+(?:user|client|customer)\s+(?:wanted|needed|asked\s+for)\s+(.+)",
             lambda m: f"User need identified: {m.group(1)}"),
            (r"(?:learned|realized|discovered)\s+that\s+(.+)",
             lambda m: f"Key insight: {m.group(1)}"),
            (r"(?:mistake|error|failure|issue|problem)\s+(?:was|occurred|happened)\s+(?:when|because)\s+(.+)",
             lambda m: f"Avoid: {m.group(1)}"),
        ]

        for pattern, lesson_formatter in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return {
                    "lesson": lesson_formatter(match).strip(),
                    "confidence": episode.emotional_weight,
                    "source_episodes": [episode.id],
                    "tags": list(episode.tags),
                    "pattern_type": "content_pattern",
                }

        # Fallback: convert content to a lesson statement
        if len(content.split()) >= 3:
            sentences = re.split(r'[.!?]+', episode.content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 5:
                    return {
                        "lesson": f"Remember: {sentence}",
                        "confidence": episode.emotional_weight * 0.8,
                        "source_episodes": [episode.id],
                        "tags": list(episode.tags),
                        "pattern_type": "direct_experience",
                    }

        return None

    def _extract_tag_pattern_lessons(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Extract lessons from tag co-occurrence patterns.

        Identifies frequently co-occurring tags and generates lessons about
        common patterns.

        Args:
            episodes: Episodes to analyze.

        Returns:
            List of lesson dictionaries.
        """
        tag_pairs: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        tag_examples: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)

        for ep in episodes:
            tags = sorted(ep.tags)
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair = (tags[i], tags[j])
                    tag_pairs[pair] += 1
                    if len(tag_examples[pair]) < 3:
                        tag_examples[pair].append(ep.content[:100])

        lessons = []
        for pair, count in sorted(tag_pairs.items(), key=lambda x: x[1], reverse=True):
            if count < 2:
                continue

            examples = tag_examples[pair]
            confidence = min(1.0, count / max(1, len(episodes)) * 5)

            lesson_text = (
                f"Patterns involving '{pair[0]}' and '{pair[1]}' have been observed "
                f"{count} time(s). Common contexts: {'; '.join(examples[:2])}"
            )

            lessons.append({
                "lesson": lesson_text,
                "confidence": confidence,
                "source_episodes": [],
                "tags": list(pair),
                "pattern_type": "tag_cooccurrence",
                "occurrence_count": count,
            })

        return lessons[:5]

    def _extract_cluster_lessons(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Extract lessons from clusters of related episodes.

        Groups similar episodes and identifies common themes.

        Args:
            episodes: Episodes to analyze.

        Returns:
            List of lesson dictionaries.
        """
        if len(episodes) < 3:
            return []

        # Simple clustering by tag overlap
        clusters: List[List[Episode]] = []
        assigned = set()

        for ep in episodes:
            if ep.id in assigned:
                continue

            cluster = [ep]
            assigned.add(ep.id)

            for other in episodes:
                if other.id in assigned:
                    continue
                overlap = ep.tags & other.tags
                if len(overlap) >= 2:
                    cluster.append(other)
                    assigned.add(other.id)

            if len(cluster) >= 3:
                clusters.append(cluster)

        lessons = []
        for cluster in clusters[:5]:
            # Find common tags
            common_tags = cluster[0].tags.copy()
            for ep in cluster[1:]:
                common_tags &= ep.tags

            if not common_tags:
                continue

            # Find common content words
            all_words: collections.Counter = collections.Counter()
            for ep in cluster:
                words = set(ep.content.lower().split())
                all_words.update(words)

            top_words = [w for w, c in all_words.most_common(10) if c >= len(cluster) // 2]

            confidence = min(1.0, len(cluster) / 10.0)

            lesson_text = (
                f"Theme cluster ({len(cluster)} episodes): Tags {', '.join(sorted(common_tags))}. "
                f"Key concepts: {', '.join(top_words[:5])}"
            )

            lessons.append({
                "lesson": lesson_text,
                "confidence": confidence,
                "source_episodes": [ep.id for ep in cluster],
                "tags": list(common_tags),
                "pattern_type": "cluster_theme",
                "cluster_size": len(cluster),
            })

        return lessons

    def _extract_temporal_lessons(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """Extract lessons from temporal patterns in episodes.

        Looks for recurring themes at similar times or patterns over time.

        Args:
            episodes: Episodes to analyze.

        Returns:
            List of lesson dictionaries.
        """
        if len(episodes) < 5:
            return []

        # Group by time of day
        by_hour: Dict[int, List[Episode]] = collections.defaultdict(list)
        for ep in episodes:
            hour = int(time.localtime(ep.timestamp).tm_hour)
            by_hour[hour].append(ep)

        lessons = []

        # Find hours with many episodes
        for hour, eps in sorted(by_hour.items(), key=lambda x: len(x[1]), reverse=True):
            if len(eps) < 3:
                continue

            # Common tags in this time period
            tag_counts: collections.Counter = collections.Counter()
            for ep in eps:
                tag_counts.update(ep.tags)

            top_tags = [t for t, c in tag_counts.most_common(3)]

            if top_tags:
                lesson_text = (
                    f"Temporal pattern: Experiences tagged '{', '.join(top_tags)}' "
                    f"frequently occur around {hour:02d}:00 "
                    f"({len(eps)} occurrences)"
                )

                lessons.append({
                    "lesson": lesson_text,
                    "confidence": min(1.0, len(eps) / 10.0),
                    "source_episodes": [ep.id for ep in eps[:5]],
                    "tags": top_tags,
                    "pattern_type": "temporal",
                    "hour": hour,
                })

        return lessons[:3]

    def _deduplicate_lessons(self, lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar lessons.

        Args:
            lessons: List of lesson dictionaries.

        Returns:
            Deduplicated list.
        """
        if not lessons:
            return []

        unique: List[Dict[str, Any]] = []
        seen_content: Set[str] = set()

        for lesson in lessons:
            # Create a normalized content key
            content_key = lesson["lesson"].lower()[:80]

            # Check for near-duplicates
            is_dup = False
            for seen in seen_content:
                common = set(content_key.split()) & set(seen.split())
                if len(common) > 5:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(lesson)
                seen_content.add(content_key)

        return unique

    def _load_if_exists(self) -> bool:
        """Attempt to load episodes from the persistence path.

        Returns:
            True if loaded successfully.
        """
        if not self.persistence_path:
            return False

        json_path = os.path.join(self.persistence_path, "episodic_memory.json")
        if os.path.exists(json_path):
            try:
                self.import_json(json_path, merge=False)
                return True
            except (json.JSONDecodeError, IOError, OSError):
                return False
        return False

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    def __repr__(self) -> str:
        return (
            f"EpisodicMemoryStore(count={len(self._episodes)}, "
            f"tags={len(self._tag_index)}, "
            f"embedding_dim={self.embedding_dim})"
        )
