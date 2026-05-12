"""
Nexus Semantic Memory
=====================
Semantic (factual) memory system for storing, querying, and managing factual knowledge.

Semantic memory stores generalized factual knowledge independent of specific
episodes or experiences. Unlike episodic memory which captures "what happened when,"
semantic memory captures "what is known" — facts, concepts, rules, and general
knowledge that persist across contexts.

Key Capabilities:
- Store facts with source attribution and confidence scores
- Retrieve relevant facts via semantic similarity
- Verify facts with evidence and update confidence
- Detect contradictions between facts
- Merge overlapping facts
- Categorize and filter facts
- Import/export to JSON

Architecture:
- **Fact**: Data class representing a single factual knowledge entry
- **SemanticMemoryStore**: Main store for managing factual knowledge
- **FactEncoder**: Encoder for fact content embeddings
- **ContradictionDetector**: Detects contradictory facts
- **FactMerger**: Merges overlapping/redundant facts
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
import threading


# ═══════════════════════════════════════════════════════════════════════════════
# Fact Data Class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fact:
    """A single factual knowledge entry in the semantic memory.

    Facts represent generalized knowledge that is independent of specific
    experiences. They are stored with confidence scores that can be updated
    based on verification with evidence.

    Attributes:
        id: Unique identifier for this fact.
        content: The factual statement or knowledge content.
        source: Provenance or origin of this fact (e.g., "wikipedia", "user_stated").
        confidence: Confidence score in [0.0, 1.0] representing how reliably
            this fact has been verified. Higher means more trustworthy.
        created_at: Unix timestamp of when the fact was first recorded.
        last_verified: Unix timestamp of the most recent verification.
        related_ids: Set of fact IDs that are semantically related.
        category: Category label for organizing facts (e.g., "science", "history").
        metadata: Additional arbitrary metadata.
        embedding: Optional tensor embedding for similarity search.
        verification_count: Number of times this fact has been verified.
        contradiction_ids: Set of fact IDs that contradict this fact.
        is_active: Whether this fact is currently active (not deprecated).
    """
    id: str = ""
    content: str = ""
    source: str = ""
    confidence: float = 0.5
    created_at: float = 0.0
    last_verified: float = 0.0
    related_ids: Set[str] = field(default_factory=set)
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None
    verification_count: int = 0
    contradiction_ids: Set[str] = field(default_factory=set)
    is_active: bool = True

    @staticmethod
    def generate_id(content: str, source: str = "", timestamp: Optional[float] = None) -> str:
        """Generate a deterministic unique ID for a fact.

        Args:
            content: Fact content string.
            source: Source of the fact.
            timestamp: Optional timestamp.

        Returns:
            SHA-256 hash truncated to 16 hex characters.
        """
        if timestamp is None:
            timestamp = time.time()
        raw = f"fact:{content}:{source}:{timestamp}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize fact to a JSON-compatible dictionary.

        Returns:
            Dictionary with all fact fields.
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
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_verified": self.last_verified,
            "related_ids": list(self.related_ids),
            "category": self.category,
            "metadata": self.metadata,
            "embedding": embedding_list,
            "verification_count": self.verification_count,
            "contradiction_ids": list(self.contradiction_ids),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        """Deserialize a fact from a dictionary.

        Args:
            data: Dictionary with fact fields.

        Returns:
            Reconstructed Fact instance.
        """
        embedding = None
        if data.get("embedding") is not None:
            embedding_list = data["embedding"]
            if isinstance(embedding_list, list):
                embedding = torch.tensor(embedding_list, dtype=torch.float32)

        related_ids = data.get("related_ids", [])
        contradiction_ids = data.get("contradiction_ids", [])

        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            confidence=data.get("confidence", 0.5),
            created_at=data.get("created_at", 0.0),
            last_verified=data.get("last_verified", 0.0),
            related_ids=set(related_ids) if isinstance(related_ids, list) else set(),
            category=data.get("category", ""),
            metadata=data.get("metadata", {}),
            embedding=embedding,
            verification_count=data.get("verification_count", 0),
            contradiction_ids=set(contradiction_ids) if isinstance(contradiction_ids, list) else set(),
            is_active=data.get("is_active", True),
        )

    def verify(self, evidence: str, new_confidence: Optional[float] = None) -> None:
        """Update this fact with verification evidence.

        Args:
            evidence: Description of the evidence supporting (or refuting) this fact.
            new_confidence: New confidence score. If None, slightly boosts confidence.
        """
        self.verification_count += 1
        self.last_verified = time.time()

        if new_confidence is not None:
            self.confidence = max(0.0, min(1.0, new_confidence))
        else:
            # Small confidence boost for each verification
            self.confidence = min(1.0, self.confidence + 0.05)

        # Store evidence in metadata
        if "evidence" not in self.metadata:
            self.metadata["evidence"] = []
        self.metadata["evidence"].append({
            "text": evidence,
            "timestamp": self.last_verified,
            "confidence_after": self.confidence,
        })

    def deprecate(self, reason: str = "") -> None:
        """Mark this fact as inactive/deprecated.

        Deprecated facts are not returned in queries but are kept for
        historical tracking and contradiction analysis.

        Args:
            reason: Reason for deprecation.
        """
        self.is_active = False
        if reason:
            self.metadata["deprecation_reason"] = reason
            self.metadata["deprecated_at"] = time.time()

    def age_hours(self) -> float:
        """Get the age of this fact in hours since creation."""
        return (time.time() - self.created_at) / 3600.0

    def confidence_level(self) -> str:
        """Get a human-readable confidence level label.

        Returns:
            One of: 'very_low', 'low', 'medium', 'high', 'very_high'.
        """
        if self.confidence < 0.2:
            return "very_low"
        elif self.confidence < 0.4:
            return "low"
        elif self.confidence < 0.6:
            return "medium"
        elif self.confidence < 0.8:
            return "high"
        else:
            return "very_high"

    def __repr__(self) -> str:
        content_preview = self.content[:40] + ("..." if len(self.content) > 40 else "")
        return (
            f"Fact(id={self.id!r}, content={content_preview!r}, "
            f"confidence={self.confidence:.2f}, category={self.category!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Fact Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class FactEncoder(nn.Module):
    """Lightweight encoder for fact content embeddings.

    Uses a character-level CNN approach that captures morphological patterns
    and keyword information:
    1. Convert text to character indices
    2. Apply character-level embedding
    3. Apply 1D convolutions to capture n-gram patterns
    4. Global max pooling
    5. Project to output dimension
    6. L2 normalize
    """

    def __init__(
        self,
        char_vocab_size: int = 128,
        char_embed_dim: int = 32,
        num_filters: int = 128,
        filter_sizes: List[int] = None,
        output_dim: int = 256,
    ):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.output_dim = output_dim

        # Character embedding
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        # Convolutional filters for different n-gram sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, fs) for fs in filter_sizes
        ])

        # Projection to output dimension
        total_filters = num_filters * len(filter_sizes)
        self.projection = nn.Sequential(
            nn.Linear(total_filters, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.char_embedding.weight[0])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _text_to_char_ids(self, text: str, max_len: int = 512) -> List[int]:
        """Convert text to character ID indices.

        Args:
            text: Input text.
            max_len: Maximum character length.

        Returns:
            List of character IDs.
        """
        ids = []
        for char in text.lower()[:max_len]:
            code = ord(char)
            if code < self.char_vocab_size:
                ids.append(code)
            else:
                ids.append(1)  # UNK
        return ids if ids else [0]

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode fact text into normalized embedding vectors.

        Args:
            text: A string or list of strings.

        Returns:
            Normalized embedding tensor.
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        # Convert to character IDs
        all_ids = [self._text_to_char_ids(t) for t in text]
        max_len = max(len(ids) for ids in all_ids) if all_ids else 1

        # Pad
        padded = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
        char_tensor = torch.tensor(padded, dtype=torch.long)

        # Embed characters: (batch, seq, char_embed_dim)
        char_embeds = self.char_embedding(char_tensor)

        # Transpose for Conv1d: (batch, char_embed_dim, seq)
        char_embeds = char_embeds.transpose(1, 2)

        # Apply convolutions and max pool
        conv_outputs = []
        for conv in self.convs:
            # (batch, num_filters, seq - filter_size + 1)
            c = torch.relu(conv(char_embeds))
            # Global max pooling: (batch, num_filters)
            c = c.max(dim=2)[0]
            conv_outputs.append(c)

        # Concatenate: (batch, total_filters)
        combined = torch.cat(conv_outputs, dim=1)

        # Project: (batch, output_dim)
        projected = self.projection(combined)

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
# Contradiction Detector
# ═══════════════════════════════════════════════════════════════════════════════

class ContradictionDetector:
    """Detects potentially contradictory facts.

    Uses a combination of heuristic rules and embedding similarity to identify
    facts that may contradict each other.

    Heuristics used:
    1. Negation detection: Facts containing "not", "no", "never", "isn't", etc.
    2. Numerical comparison: Facts with the same subject but different values
    3. Mutual exclusion: Facts with explicit exclusion markers
    4. Embedding similarity: High similarity with negation markers suggests contradiction
    """

    # Negation patterns that indicate potential contradiction
    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bisn'?t\b", r"\baren'?t\b",
        r"\bwasn'?t\b", r"\bweren'?t\b", r"\bdon'?t\b", r"\bdoesn'?t\b",
        r"\bdidn'?t\b", r"\bwon'?t\b", r"\bcan'?t\b", r"\bcouldn'?t\b",
        r"\bshouldn'?t\b", r"\bwouldn'?t\b", r"\bhasn'?t\b", r"\bhaven'?t\b",
        r"\bhadn'?t\b", r"\bneither\b", r"\bnor\b", r"\bwithout\b",
        r"\bimpossible\b", r"\bfalse\b", r"\bincorrect\b", r"\buntrue\b",
        r"\bcontradict\b", r"\bdisagree\b", r"\bdisprove\b", r"\brefute\b",
    ]

    # Quantitative comparison patterns
    QUANTITY_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(?:percent|%|degrees?|years?|meters?|km|miles?|kg|pounds?|feet?|inches?)",
        r"(?:is|are|was|were|equals?|approximately)\s+(\d+(?:\.\d+)?)",
    ]

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self._negation_re = [re.compile(p, re.IGNORECASE) for p in self.NEGATION_PATTERNS]
        self._quantity_re = [re.compile(p, re.IGNORECASE) for p in self.QUANTITY_PATTERNS]

    def has_negation(self, text: str) -> bool:
        """Check if text contains negation patterns.

        Args:
            text: Input text.

        Returns:
            True if negation patterns are found.
        """
        for pattern_re in self._negation_re:
            if pattern_re.search(text):
                return True
        return False

    def extract_quantities(self, text: str) -> List[float]:
        """Extract numerical quantities from text.

        Args:
            text: Input text.

        Returns:
            List of extracted numerical values.
        """
        quantities = []
        for pattern_re in self._quantity_re:
            matches = pattern_re.findall(text)
            for match in matches:
                try:
                    quantities.append(float(match))
                except (ValueError, TypeError):
                    continue
        return quantities

    def detect_contradiction(self, fact1: Fact, fact2: Fact) -> Tuple[bool, float]:
        """Detect if two facts potentially contradict each other.

        Uses multiple heuristics to assess contradiction likelihood.

        Args:
            fact1: First fact.
            fact2: Second fact.

        Returns:
            Tuple of (is_contradiction, confidence_score).
        """
        score = 0.0

        # Check negation patterns
        f1_negated = self.has_negation(fact1.content)
        f2_negated = self.has_negation(fact2.content)

        if f1_negated != f2_negated:
            # One is negated, the other isn't
            # Check if they share significant vocabulary
            words1 = set(re.findall(r'\b\w{4,}\b', fact1.content.lower()))
            words2 = set(re.findall(r'\b\w{4,}\b', fact2.content.lower()))
            common = words1 & words2

            if len(common) >= 2:
                # Significant word overlap + differing negation → likely contradiction
                overlap_ratio = len(common) / max(1, min(len(words1), len(words2)))
                if overlap_ratio >= 0.3:
                    score += 0.6 * overlap_ratio

        # Check quantity conflicts
        quantities1 = self.extract_quantities(fact1.content)
        quantities2 = self.extract_quantities(fact2.content)

        if quantities1 and quantities2:
            # Check if quantities refer to the same thing but differ significantly
            for q1 in quantities1:
                for q2 in quantities2:
                    if q1 > 0 and q2 > 0:
                        ratio = max(q1, q2) / min(q1, q2)
                        if ratio > 1.5:
                            # Check if subjects match
                            words1 = set(re.findall(r'\b\w+\b', fact1.content.lower()))
                            words2 = set(re.findall(r'\b\w+\b', fact2.content.lower()))
                            if len(words1 & words2) >= 3:
                                score += 0.4

        # Check for explicit contradiction markers
        contradiction_markers = [
            "however", "but", "although", "despite", "contrary to",
            "in contrast", "on the other hand", "nevertheless",
        ]
        combined = (fact1.content + " " + fact2.content).lower()
        marker_count = sum(1 for m in contradiction_markers if m in combined)
        if marker_count >= 2:
            score += 0.2

        # Embedding-based check (if available)
        if fact1.embedding is not None and fact2.embedding is not None:
            similarity = float(torch.dot(fact1.embedding, fact2.embedding).clamp(-1, 1))
            # High similarity between one negated and one non-negated
            if similarity > self.similarity_threshold and (f1_negated != f2_negated):
                score += 0.3

        is_contradiction = score >= 0.5
        return is_contradiction, min(1.0, score)


# ═══════════════════════════════════════════════════════════════════════════════
# Fact Merger
# ═══════════════════════════════════════════════════════════════════════════════

class FactMerger:
    """Merges overlapping or redundant facts into a single consolidated fact.

    The merger identifies facts that convey the same or very similar information
    and combines them, preserving the most confident information from each source.
    """

    def merge_facts(self, facts: List[Fact]) -> Optional[Fact]:
        """Merge multiple related facts into a single consolidated fact.

        The merge strategy:
        1. Use the most confident fact's content as the base
        2. Combine metadata from all facts
        3. Take the maximum confidence (capped at 1.0 with a small boost)
        4. Combine all source attributions
        5. Combine all categories
        6. Merge all evidence from verifications

        Args:
            facts: List of facts to merge (must be at least 2).

        Returns:
            Merged Fact, or None if fewer than 2 facts provided.
        """
        if len(facts) < 2:
            return None

        # Sort by confidence descending
        sorted_facts = sorted(facts, key=lambda f: f.confidence, reverse=True)
        primary = sorted_facts[0]

        # Create merged fact based on primary
        merged = copy.deepcopy(primary)

        # Combine sources
        sources = set()
        all_evidence = []
        all_categories = set()

        for fact in sorted_facts:
            if fact.source:
                sources.add(fact.source)
            if fact.category:
                all_categories.add(fact.category)
            if "evidence" in fact.metadata and isinstance(fact.metadata["evidence"], list):
                all_evidence.extend(fact.metadata["evidence"])

        merged.source = "; ".join(sorted(sources)) if sources else primary.source
        merged.category = ", ".join(sorted(all_categories)) if all_categories else primary.category

        # Boost confidence for multi-source confirmation
        merged.confidence = min(1.0, primary.confidence + 0.05 * (len(facts) - 1))
        merged.verification_count = sum(f.verification_count for f in sorted_facts)

        # Store merge information in metadata
        merged.metadata["merged_from"] = [f.id for f in sorted_facts]
        merged.metadata["merge_count"] = len(facts)

        if all_evidence:
            merged.metadata["evidence"] = all_evidence

        # Regenerate ID
        merged.id = Fact.generate_id(merged.content, merged.source, time.time())

        return merged

    def find_merge_candidates(
        self,
        facts: List[Fact],
        similarity_threshold: float = 0.85,
    ) -> List[List[str]]:
        """Find groups of facts that should be merged.

        Groups facts by high semantic similarity (overlapping content).

        Args:
            facts: List of facts to analyze.
            similarity_threshold: Minimum similarity to consider for merging.

        Returns:
            List of fact ID groups that are candidates for merging.
        """
        if len(facts) < 2:
            return []

        groups: List[List[str]] = []
        assigned: Set[str] = set()

        # Compute pairwise similarities
        for i, f1 in enumerate(facts):
            if f1.id in assigned:
                continue

            group = [f1.id]

            for j, f2 in enumerate(facts):
                if f2.id in assigned or f1.id == f2.id:
                    continue

                similarity = self._content_similarity(f1.content, f2.content)
                if similarity >= similarity_threshold:
                    group.append(f2.id)

            if len(group) >= 2:
                groups.append(group)
                assigned.update(group)

        return groups

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Compute word-level similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Jaccard-like similarity score in [0.0, 1.0].
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Memory Store
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticMemoryStore:
    """Store and query factual (semantic) knowledge.

    The SemanticMemoryStore manages a collection of Fact objects with support
    for semantic similarity retrieval, fact verification, contradiction detection,
    fact merging, and categorical filtering.

    Features:
    - Store facts with source, confidence, and category
    - Retrieve relevant facts via embedding similarity
    - Verify facts with evidence and update confidence
    - Detect contradictions between facts
    - Merge overlapping facts
    - Filter by category
    - Comprehensive statistics

    Args:
        embedding_dim: Dimensionality of fact embeddings.
        max_facts: Maximum number of facts to store.
        persistence_path: Optional path for persistence.
        auto_verify: Whether to automatically check new facts against existing ones.

    Example:
        >>> store = SemanticMemoryStore(embedding_dim=256)
        >>> store.store_fact(
        ...     content="Python 3.12 was released in October 2023",
        ...     source="python.org",
        ...     confidence=0.95,
        ...     category="release_info",
        ... )
        >>> results = store.query_facts("When was Python 3.12 released?", top_k=3)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        max_facts: int = 100000,
        persistence_path: Optional[str] = None,
        auto_verify: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.max_facts = max_facts
        self.persistence_path = persistence_path
        self.auto_verify = auto_verify

        # Fact storage
        self._facts: Dict[str, Fact] = collections.OrderedDict()

        # Category index
        self._category_index: Dict[str, Set[str]] = collections.defaultdict(set)

        # Source index
        self._source_index: Dict[str, Set[str]] = collections.defaultdict(set)

        # Encoder
        self._encoder = FactEncoder(output_dim=embedding_dim)
        self._encoder.eval()

        # Detectors
        self._contradiction_detector = ContradictionDetector()
        self._fact_merger = FactMerger()

        # Statistics
        self._total_stored = 0
        self._total_queries = 0
        self._total_verifications = 0
        self._total_contradictions_found = 0
        self._total_merges = 0

        self._lock = threading.RLock()

        # Load from disk
        if persistence_path:
            os.makedirs(persistence_path, exist_ok=True)
            self._load_if_exists()

    def store_fact(
        self,
        content: str,
        source: str = "",
        confidence: float = 0.5,
        category: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        fact_id: Optional[str] = None,
    ) -> Fact:
        """Store a new fact in the semantic memory.

        Creates an embedding for the content, checks for contradictions with
        existing facts (if auto_verify is enabled), and stores the fact.

        Args:
            content: The factual statement.
            source: Provenance or origin of this fact.
            confidence: Initial confidence score in [0.0, 1.0].
            category: Category label for organization.
            metadata: Additional metadata.
            fact_id: Explicit ID. Auto-generated if None.

        Returns:
            The newly created Fact.

        Raises:
            ValueError: If content is empty.
        """
        if not content or not content.strip():
            raise ValueError("Fact content cannot be empty")

        content = content.strip()
        confidence = max(0.0, min(1.0, confidence))
        category = category.strip().lower() if category else ""
        metadata = metadata or {}

        with self._lock:
            # Check capacity
            if len(self._facts) >= self.max_facts:
                self._evict_low_confidence(1)

            # Generate ID
            now = time.time()
            if fact_id is None:
                fact_id = Fact.generate_id(content, source, now)

            # Generate embedding
            embedding = self._encoder.encode(content)

            # Create fact
            fact = Fact(
                id=fact_id,
                content=content,
                source=source,
                confidence=confidence,
                created_at=now,
                last_verified=now,
                category=category,
                metadata=metadata,
                embedding=embedding,
            )

            # Check for contradictions with existing facts
            if self.auto_verify:
                contradictions = self.find_contradictions_raw(fact)
                for existing_id, score in contradictions:
                    fact.contradiction_ids.add(existing_id)
                    if existing_id in self._facts:
                        self._facts[existing_id].contradiction_ids.add(fact_id)
                        self._total_contradictions_found += 1

            # Store
            self._facts[fact_id] = fact

            # Update indices
            if category:
                self._category_index[category].add(fact_id)
            if source:
                self._source_index[source].add(fact_id)

            self._total_stored += 1

            return fact

    def query_facts(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.3,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Tuple[Fact, float]]:
        """Retrieve facts relevant to a query.

        Encodes the query and computes cosine similarity against all active facts.
        Supports optional filtering by category and minimum confidence.

        Args:
            query: Query text.
            top_k: Maximum number of results.
            threshold: Minimum similarity score.
            category: Optional category filter.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of (Fact, similarity_score) tuples sorted by descending similarity.
        """
        if not query or not self._facts:
            return []

        with self._lock:
            # Encode query
            query_embedding = self._encoder.encode(query)

            # Compute similarities
            results: List[Tuple[Fact, float]] = []
            for fact in self._facts.values():
                # Skip inactive facts
                if not fact.is_active:
                    continue

                # Category filter
                if category and fact.category != category.strip().lower():
                    continue

                # Confidence filter
                if fact.confidence < min_confidence:
                    continue

                if fact.embedding is None:
                    continue

                sim = float(torch.dot(query_embedding, fact.embedding).clamp(-1, 1))
                if sim >= threshold:
                    results.append((fact, sim))

            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)

            self._total_queries += 1
            return results[:top_k]

    def verify_fact(
        self,
        fact_id: str,
        evidence: str,
        new_confidence: Optional[float] = None,
    ) -> Optional[Fact]:
        """Verify a fact with supporting evidence.

        Updates the fact's confidence, verification count, and timestamp.
        Stores the evidence in the fact's metadata.

        Args:
            fact_id: ID of the fact to verify.
            evidence: Description of the evidence.
            new_confidence: New confidence score. If None, slightly boosts confidence.

        Returns:
            Updated Fact, or None if fact_id not found.
        """
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None:
                return None

            fact.verify(evidence, new_confidence)
            self._total_verifications += 1

            return fact

    def find_contradictions(self, fact_id: str) -> List[Tuple[Fact, float]]:
        """Find facts that contradict a given fact.

        Args:
            fact_id: ID of the fact to check.

        Returns:
            List of (contradicting_fact, confidence_score) tuples.
        """
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None:
                return []

            results: List[Tuple[Fact, float]] = []
            for other_id in fact.contradiction_ids:
                other = self._facts.get(other_id)
                if other is not None and other.is_active:
                    is_contra, score = self._contradiction_detector.detect_contradiction(fact, other)
                    if is_contra:
                        results.append((other, score))

            return results

    def find_contradictions_raw(self, fact: Fact) -> List[Tuple[str, float]]:
        """Check a (potentially new) fact against all existing facts.

        Args:
            fact: Fact to check (may not be stored yet).

        Returns:
            List of (existing_fact_id, contradiction_score) tuples.
        """
        results: List[Tuple[str, float]] = []
        for existing in self._facts.values():
            if not existing.is_active:
                continue
            is_contra, score = self._contradiction_detector.detect_contradiction(fact, existing)
            if is_contra:
                results.append((existing.id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_all_contradictions(self) -> List[Tuple[Fact, Fact, float]]:
        """Scan all active facts for contradictions.

        This is an O(n²) operation and should be used sparingly.

        Returns:
            List of (fact1, fact2, score) tuples for all detected contradictions.
        """
        with self._lock:
            active_facts = [f for f in self._facts.values() if f.is_active]
            contradictions: List[Tuple[Fact, Fact, float]] = []

            checked = set()
            for f1 in active_facts:
                for f2 in active_facts:
                    if f1.id >= f2.id:
                        continue
                    pair_key = (f1.id, f2.id)
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    is_contra, score = self._contradiction_detector.detect_contradiction(f1, f2)
                    if is_contra:
                        contradictions.append((f1, f2, score))

            contradictions.sort(key=lambda x: x[2], reverse=True)
            return contradictions

    def merge_facts(self, fact_ids: List[str]) -> Optional[Fact]:
        """Merge multiple facts into a single consolidated fact.

        The original facts are deprecated and replaced by the merged fact.

        Args:
            fact_ids: List of fact IDs to merge.

        Returns:
            The merged Fact, or None if merge failed.
        """
        with self._lock:
            if len(fact_ids) < 2:
                return None

            facts_to_merge = []
            for fid in fact_ids:
                fact = self._facts.get(fid)
                if fact is not None and fact.is_active:
                    facts_to_merge.append(fact)

            if len(facts_to_merge) < 2:
                return None

            # Merge
            merged = self._fact_merger.merge_facts(facts_to_merge)
            if merged is None:
                return None

            # Deprecate originals
            for fact in facts_to_merge:
                fact.deprecate(f"Merged into fact {merged.id}")

            # Store merged fact
            self._facts[merged.id] = merged

            # Update indices
            if merged.category:
                self._category_index[merged.category].add(merged.id)

            self._total_merges += 1

            return merged

    def auto_merge(self, similarity_threshold: float = 0.85) -> int:
        """Automatically find and merge duplicate facts.

        Args:
            similarity_threshold: Similarity threshold for considering facts as duplicates.

        Returns:
            Number of merges performed.
        """
        with self._lock:
            active_facts = [f for f in self._facts.values() if f.is_active]
            groups = self._fact_merger.find_merge_candidates(active_facts, similarity_threshold)

            merges = 0
            for group in groups:
                result = self.merge_facts(group)
                if result is not None:
                    merges += 1

            return merges

    def get_by_category(self, category: str) -> List[Fact]:
        """Get all facts in a specific category.

        Args:
            category: Category label to filter by.

        Returns:
            List of active facts in the category.
        """
        category_normalized = category.strip().lower()
        fact_ids = self._category_index.get(category_normalized, set())

        facts = []
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact is not None and fact.is_active:
                facts.append(fact)

        return facts

    def get_by_source(self, source: str) -> List[Fact]:
        """Get all facts from a specific source.

        Args:
            source: Source label to filter by.

        Returns:
            List of active facts from the source.
        """
        source_ids = self._source_index.get(source, set())

        facts = []
        for fid in source_ids:
            fact = self._facts.get(fid)
            if fact is not None and fact.is_active:
                facts.append(fact)

        return facts

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a specific fact by ID.

        Args:
            fact_id: ID of the fact.

        Returns:
            Fact if found, None otherwise.
        """
        return self._facts.get(fact_id)

    def update_fact(
        self,
        fact_id: str,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Fact]:
        """Update an existing fact.

        Args:
            fact_id: ID of the fact to update.
            content: New content (triggers embedding recompute).
            confidence: New confidence score.
            category: New category.
            metadata: New metadata (merged with existing).

        Returns:
            Updated Fact, or None if not found.
        """
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None:
                return None

            old_category = fact.category

            if content is not None and content.strip():
                # Remove from old category index
                if old_category and old_category in self._category_index:
                    self._category_index[old_category].discard(fact_id)

                fact.content = content.strip()
                fact.embedding = self._encoder.encode(fact.content)

            if confidence is not None:
                fact.confidence = max(0.0, min(1.0, confidence))

            if category is not None:
                new_category = category.strip().lower()
                if old_category and old_category in self._category_index:
                    self._category_index[old_category].discard(fact_id)
                fact.category = new_category
                if new_category:
                    self._category_index[new_category].add(fact_id)

            if metadata is not None:
                fact.metadata.update(metadata)

            return fact

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID.

        Args:
            fact_id: ID of the fact to delete.

        Returns:
            True if found and deleted.
        """
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None:
                return False

            # Remove from indices
            if fact.category and fact.category in self._category_index:
                self._category_index[fact.category].discard(fact_id)
            if fact.source and fact.source in self._source_index:
                self._source_index[fact.source].discard(fact_id)

            # Remove contradiction links
            for contra_id in fact.contradiction_ids:
                if contra_id in self._facts:
                    self._facts[contra_id].contradiction_ids.discard(fact_id)

            del self._facts[fact_id]
            return True

    def count(self) -> int:
        """Return the number of active facts.

        Returns:
            Count of active facts.
        """
        return sum(1 for f in self._facts.values() if f.is_active)

    def count_all(self) -> int:
        """Return the total number of facts including deprecated ones.

        Returns:
            Total fact count.
        """
        return len(self._facts)

    def get_all_categories(self) -> Dict[str, int]:
        """Get all categories and their active fact counts.

        Returns:
            Dictionary mapping categories to counts.
        """
        result: Dict[str, int] = collections.defaultdict(int)
        for fact in self._facts.values():
            if fact.is_active and fact.category:
                result[fact.category] += 1
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_all_sources(self) -> Dict[str, int]:
        """Get all sources and their active fact counts.

        Returns:
            Dictionary mapping sources to counts.
        """
        result: Dict[str, int] = collections.defaultdict(int)
        for fact in self._facts.values():
            if fact.is_active and fact.source:
                result[fact.source] += 1
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive statistics about the semantic memory store.

        Returns:
            Dictionary with store statistics.
        """
        active_facts = [f for f in self._facts.values() if f.is_active]
        all_facts = list(self._facts.values())

        if not all_facts:
            return {
                "total_facts": 0,
                "active_facts": 0,
                "max_facts": self.max_facts,
                "total_stored": self._total_stored,
                "total_queries": self._total_queries,
                "total_verifications": self._total_verifications,
                "total_contradictions": self._total_contradictions_found,
                "total_merges": self._total_merges,
            }

        active_confidences = [f.confidence for f in active_facts]
        all_confidences = [f.confidence for f in all_facts]

        # Confidence distribution
        confidence_buckets = {"very_high": 0, "high": 0, "medium": 0, "low": 0, "very_low": 0}
        for c in active_confidences:
            level = Fact(id="", content="", confidence=c).confidence_level()
            confidence_buckets[level] += 1

        return {
            "total_facts": len(all_facts),
            "active_facts": len(active_facts),
            "deprecated_facts": len(all_facts) - len(active_facts),
            "max_facts": self.max_facts,
            "utilization": len(all_facts) / self.max_facts,
            "total_stored": self._total_stored,
            "total_queries": self._total_queries,
            "total_verifications": self._total_verifications,
            "total_contradictions": self._total_contradictions_found,
            "total_merges": self._total_merges,
            "avg_confidence": sum(active_confidences) / len(active_confidences) if active_confidences else 0.0,
            "min_confidence": min(active_confidences) if active_confidences else 0.0,
            "max_confidence": max(active_confidences) if active_confidences else 0.0,
            "confidence_distribution": confidence_buckets,
            "categories": len(self._category_index),
            "sources": len(self._source_index),
            "top_categories": list(self.get_all_categories().items())[:10],
            "top_sources": list(self.get_all_sources().items())[:10],
        }

    def export_json(self, path: str) -> None:
        """Export all facts to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "facts": [f.to_dict() for f in self._facts.values()],
            "config": {
                "embedding_dim": self.embedding_dim,
                "max_facts": self.max_facts,
                "auto_verify": self.auto_verify,
            },
            "statistics": self.get_statistics(),
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def import_json(self, path: str, merge: bool = True) -> int:
        """Import facts from a JSON file.

        Args:
            path: Input file path.
            merge: If True, merges with existing facts. If False, replaces all.

        Returns:
            Number of facts imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with self._lock:
            if not merge:
                self._facts.clear()
                self._category_index.clear()
                self._source_index.clear()

            count = 0
            for fact_data in data.get("facts", []):
                fact = Fact.from_dict(fact_data)

                # Regenerate embedding if needed
                if fact.embedding is None and fact.content:
                    fact.embedding = self._encoder.encode(fact.content)

                self._facts[fact.id] = fact

                if fact.category:
                    self._category_index[fact.category].add(fact.id)
                if fact.source:
                    self._source_index[fact.source].add(fact.id)

                count += 1

            self._total_stored += count
            return count

    def clear(self) -> int:
        """Remove all facts from the store.

        Returns:
            Number of facts removed.
        """
        with self._lock:
            count = len(self._facts)
            self._facts.clear()
            self._category_index.clear()
            self._source_index.clear()
            return count

    def search_exact(self, content: str, case_sensitive: bool = False) -> List[Fact]:
        """Search for facts with exact content matches.

        Args:
            content: Exact content to search for.
            case_sensitive: Whether to match case.

        Returns:
            List of matching facts.
        """
        results = []
        search_content = content if case_sensitive else content.lower()

        for fact in self._facts.values():
            fact_content = fact.content if case_sensitive else fact.content.lower()
            if fact_content == search_content:
                results.append(fact)

        return results

    def search_substring(self, query: str, case_sensitive: bool = False) -> List[Fact]:
        """Search for facts containing a substring.

        Args:
            query: Substring to search for.
            case_sensitive: Whether to match case.

        Returns:
            List of matching facts.
        """
        results = []
        search_query = query if case_sensitive else query.lower()

        for fact in self._facts.values():
            fact_content = fact.content if case_sensitive else fact.content.lower()
            if search_query in fact_content:
                results.append(fact)

        return results

    def get_high_confidence(self, threshold: float = 0.8, category: Optional[str] = None) -> List[Fact]:
        """Get facts with confidence above a threshold.

        Args:
            threshold: Minimum confidence.
            category: Optional category filter.

        Returns:
            List of high-confidence facts.
        """
        results = []
        for fact in self._facts.values():
            if not fact.is_active:
                continue
            if fact.confidence >= threshold:
                if category is None or fact.category == category.strip().lower():
                    results.append(fact)

        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def _evict_low_confidence(self, count: int) -> int:
        """Evict the lowest confidence facts to free capacity.

        Args:
            count: Number of facts to evict.

        Returns:
            Number evicted.
        """
        evictable = [
            (fid, fact) for fid, fact in self._facts.items()
            if fact.is_active
        ]
        evictable.sort(key=lambda x: x[1].confidence)

        evicted = 0
        for fid, fact in evictable[:count]:
            self.delete_fact(fid)
            evicted += 1

        return evicted

    def _load_if_exists(self) -> bool:
        """Attempt to load from persistence path.

        Returns:
            True if loaded successfully.
        """
        if not self.persistence_path:
            return False

        json_path = os.path.join(self.persistence_path, "semantic_memory.json")
        if os.path.exists(json_path):
            try:
                self.import_json(json_path, merge=False)
                return True
            except (json.JSONDecodeError, IOError, OSError):
                return False
        return False

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def __repr__(self) -> str:
        return (
            f"SemanticMemoryStore(count={self.count()}/{self.max_facts}, "
            f"categories={len(self._category_index)}, "
            f"contradictions={self._total_contradictions_found})"
        )
