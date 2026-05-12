"""
Data Curation and Strategy for Nexus LLM.

Provides comprehensive data management for LLM training including curation,
balancing, deduplication, quality scoring, mixing, budget management,
and streaming data loading.

Classes:
    DataCurator: Select and order training data for optimal learning.
    DomainBalancer: Balance training data across domains.
    DifficultySampler: Sample data by difficulty (easy to hard).
    DataDeduplication: MinHash + LSH based deduplication.
    QualityScorer: Score data quality (perplexity, coherence, diversity).
    DataMixer: Mix multiple data sources with configurable ratios.
    TokenBudgetManager: Manage total token budget across training.
    ReplayBuffer: Store and replay important examples.
    DataPipeline: End-to-end data processing pipeline.
    StreamingDataLoader: Load data in streaming fashion from disk.
"""

from __future__ import annotations

import abc
import hashlib
import heapq
import json
import logging
import math
import os
import random
import re
import struct
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class Domain(Enum):
    """Training data domains."""
    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    SCIENCE = "science"
    HUMANITIES = "humanities"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"


class QualityDimension(Enum):
    """Dimensions of data quality."""
    COHERENCE = auto()
    DIVERSITY = auto()
    INFORMATIVENESS = auto()
    FLUENCY = auto()
    ACCURACY = auto()
    COMPLETENESS = auto()


class DedupMethod(Enum):
    """Deduplication method."""
    EXACT = auto()
    MINHASH_LSH = auto()
    FUZZY = auto()
    SEMANTIC = auto()


class SamplingStrategy(Enum):
    """Data sampling strategy."""
    UNIFORM = auto()
    WEIGHTED = auto()
    TEMPERATURE = auto()
    CURRICULUM = auto()
    ADVERSARIAL = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DataSample:
    """A single training data sample with metadata."""
    text: str
    tokens: Optional[List[int]] = None
    domain: str = "general"
    quality_score: float = 0.5
    difficulty_score: float = 0.5
    sample_id: Optional[str] = None
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    times_seen: int = 0
    last_seen_epoch: int = 0
    weight: float = 1.0
    is_active: bool = True

    def __post_init__(self):
        if self.sample_id is None:
            self.sample_id = hashlib.md5(self.text.encode()).hexdigest()[:12]
        if self.token_count == 0 and self.tokens:
            self.token_count = len(self.tokens)
        elif self.token_count == 0:
            self.token_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "domain": self.domain,
            "quality_score": self.quality_score,
            "difficulty_score": self.difficulty_score,
            "source": self.source,
            "token_count": self.token_count,
            "times_seen": self.times_seen,
            "weight": self.weight,
            "is_active": self.is_active,
        }


@dataclass
class DataSource:
    """A data source with samples and metadata."""
    name: str
    samples: List[DataSample] = field(default_factory=list)
    domain: str = "general"
    weight: float = 1.0
    priority: int = 0
    quality_threshold: float = 0.3
    max_samples: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_samples(self) -> List[DataSample]:
        """Return samples that pass quality threshold."""
        if self.max_samples > 0:
            active = [s for s in self.samples if s.is_active and s.quality_score >= self.quality_threshold]
            return active[:self.max_samples]
        return [s for s in self.samples if s.is_active and s.quality_score >= self.quality_threshold]

    def total_tokens(self) -> int:
        """Return total tokens in this source."""
        return sum(s.token_count for s in self.effective_samples())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "weight": self.weight,
            "priority": self.priority,
            "num_samples": len(self.samples),
            "effective_samples": len(self.effective_samples()),
            "total_tokens": self.total_tokens(),
            "quality_threshold": self.quality_threshold,
        }


@dataclass
class MixingConfig:
    """Configuration for data mixing."""
    source_ratios: Dict[str, float] = field(default_factory=dict)
    domain_ratios: Dict[str, float] = field(default_factory=dict)
    quality_threshold: float = 0.3
    min_quality: float = 0.1
    difficulty_mode: SamplingStrategy = SamplingStrategy.UNIFORM
    temperature: float = 1.0
    shuffle: bool = True
    seed: int = 42


@dataclass
class BudgetAllocation:
    """Token budget allocation for a data source."""
    source_name: str
    allocated_tokens: int
    used_tokens: int = 0
    remaining_tokens: int = 0

    def __post_init__(self):
        self.remaining_tokens = self.allocated_tokens

    @property
    def utilization(self) -> float:
        return self.used_tokens / max(1, self.allocated_tokens)

    def consume(self, tokens: int):
        """Consume tokens from the allocation."""
        self.used_tokens += tokens
        self.remaining_tokens -= tokens


# ---------------------------------------------------------------------------
# Quality Scorer
# ---------------------------------------------------------------------------

class QualityScorer:
    """Score data quality across multiple dimensions.

    Evaluates text quality based on coherence, diversity, informativeness,
    fluency, and completeness.

    Args:
        dimension_weights: Weights for each quality dimension.
        vocab_frequencies: Optional vocabulary frequency counter.
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None,
        vocab_frequencies: Optional[Counter] = None,
    ):
        self.dimension_weights = dimension_weights or {
            "coherence": 0.25,
            "diversity": 0.20,
            "informativeness": 0.20,
            "fluency": 0.15,
            "completeness": 0.20,
        }
        self.vocab_frequencies = vocab_frequencies or Counter()
        self._total_vocab = sum(self.vocab_frequencies.values()) if self.vocab_frequencies else 1

    def score(self, text: str) -> float:
        """Compute overall quality score for text.

        Args:
            text: Input text.

        Returns:
            Quality score in [0, 1].
        """
        if not text or not text.strip():
            return 0.0

        scores = {
            "coherence": self._score_coherence(text),
            "diversity": self._score_diversity(text),
            "informativeness": self._score_informativeness(text),
            "fluency": self._score_fluency(text),
            "completeness": self._score_completeness(text),
        }

        composite = 0.0
        total_weight = 0.0
        for dim, score in scores.items():
            w = self.dimension_weights.get(dim, 0.0)
            composite += w * score
            total_weight += w

        return max(0.0, min(1.0, composite / max(1e-8, total_weight)))

    def score_detailed(self, text: str) -> Dict[str, float]:
        """Compute and return all individual quality scores."""
        return {
            "coherence": self._score_coherence(text),
            "diversity": self._score_diversity(text),
            "informativeness": self._score_informativeness(text),
            "fluency": self._score_fluency(text),
            "completeness": self._score_completeness(text),
            "overall": self.score(text),
        }

    def _score_coherence(self, text: str) -> float:
        """Score text coherence."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.7

        connector_words = {
            "because", "therefore", "however", "also", "moreover",
            "furthermore", "additionally", "although", "while",
            "since", "thus", "hence", "consequently", "nevertheless",
            "meanwhile", "similarly", "conversely", "instead",
            "for example", "in contrast", "as a result", "in addition",
            "first", "second", "third", "finally", "overall",
        }

        connector_count = 0
        total_words = 0
        for sent in sentences:
            words = sent.lower().split()
            total_words += len(words)
            connector_count += sum(1 for w in words if w in connector_words)

        connector_ratio = connector_count / max(1, total_words)
        score = 0.5 + connector_ratio * 3.0

        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            avg_len = sum(sentence_lengths) / len(sentence_lengths)
            length_var = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            consistency = max(0, 1.0 - length_var / max(1, avg_len ** 2))
            score = 0.7 * score + 0.3 * consistency

        return min(1.0, score)

    def _score_diversity(self, text: str) -> float:
        """Score vocabulary diversity."""
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)
        type_token_ratio = unique_words / total_words

        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]}_{words[i+1]}")
        unique_bigrams = len(set(bigrams))
        bigram_ratio = unique_bigrams / max(1, len(bigrams))

        score = 0.5 * type_token_ratio + 0.3 * bigram_ratio + 0.2

        if total_words > 50:
            first_half = words[:total_words // 2]
            second_half = words[total_words // 2:]
            overlap = set(first_half) & set(second_half)
            cross_ratio = 1.0 - len(overlap) / max(1, len(set(first_half) | set(second_half)))
            score = 0.7 * score + 0.3 * cross_ratio

        return min(1.0, score)

    def _score_informativeness(self, text: str) -> float:
        """Score informativeness of text."""
        words = text.lower().split()
        if not words:
            return 0.0

        content_words = sum(
            1 for w in words
            if w not in {"the", "a", "an", "is", "are", "was", "were",
                         "be", "been", "being", "have", "has", "had",
                         "do", "does", "did", "will", "would", "could",
                         "should", "may", "might", "can", "shall",
                         "to", "of", "in", "for", "on", "with", "at",
                         "by", "from", "as", "into", "through", "during",
                         "before", "after", "above", "below", "between"}
        )
        content_ratio = content_words / max(1, len(words))

        numbers = sum(1 for w in words if any(c.isdigit() for c in w))
        number_ratio = numbers / max(1, len(words))

        technical_words = sum(
            1 for w in words
            if len(w) > 8 or w.endswith("tion") or w.endswith("ment")
            or w.endswith("ness") or w.endswith("ity") or w.endswith("ism")
        )
        tech_ratio = technical_words / max(1, len(words))

        score = 0.4 * content_ratio + 0.2 * number_ratio + 0.2 * tech_ratio

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
            density = min(1.0, avg_sent_len / 25.0)
            score = 0.8 * score + 0.2 * density

        return min(1.0, score)

    def _score_fluency(self, text: str) -> float:
        """Score text fluency."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        issues = 0
        total_words = 0

        for sent in sentences:
            words = sent.split()
            total_words += len(words)

            if len(words) > 50:
                issues += 1

            repeated_bigrams = 0
            for i in range(len(words) - 1):
                for j in range(i + 2, len(words) - 1):
                    if words[i] == words[j] and words[i + 1] == words[j + 1]:
                        repeated_bigrams += 1
                        break
            issues += repeated_bigrams

        if total_words > 0:
            issue_rate = issues / max(1, total_words)
            score = 1.0 - min(1.0, issue_rate * 10)
        else:
            score = 0.0

        avg_word_len = sum(len(w) for w in text.split()) / max(1, len(text.split()))
        if avg_word_len > 2:
            score *= min(1.0, avg_word_len / 5.0)

        return max(0.0, min(1.0, score))

    def _score_completeness(self, text: str) -> float:
        """Score text completeness."""
        text_stripped = text.strip()

        if not text_stripped:
            return 0.0

        incomplete_patterns = [
            r'\b\w+ \.\.\.$',
            r'etc\.?\s*$',
            r'and so on\.?\s*$',
            r'\?$',
            r'\bTODO\b',
            r'\bN/A\b',
            r'\bplaceholder\b',
        ]
        has_incomplete = any(
            re.search(p, text_stripped, re.I) for p in incomplete_patterns
        )

        has_start = bool(text_stripped[0].isupper() or text_stripped[0] == '"')
        has_end = text_stripped[-1] in '.!?"\')'

        word_count = len(text_stripped.split())
        length_score = min(1.0, word_count / 50.0)

        score = length_score
        if has_start:
            score += 0.1
        if has_end:
            score += 0.1
        if has_incomplete:
            score -= 0.3

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Data Curator
# ---------------------------------------------------------------------------

class DataCurator:
    """Select and order training data for optimal learning.

    Analyzes data quality, difficulty, and domain distribution to
    curate an optimal training dataset.

    Args:
        quality_scorer: Optional QualityScorer instance.
        seed: Random seed.
    """

    def __init__(
        self,
        quality_scorer: Optional[QualityScorer] = None,
        seed: int = 42,
    ):
        self.quality_scorer = quality_scorer or QualityScorer()
        self.rng = random.Random(seed)
        self._sources: Dict[str, DataSource] = {}
        self._all_samples: List[DataSample] = []
        self._domain_index: Dict[str, List[int]] = defaultdict(list)

    def add_source(self, source: DataSource):
        """Add a data source."""
        self._sources[source.name] = source
        start_idx = len(self._all_samples)
        self._all_samples.extend(source.samples)
        for i, sample in enumerate(source.samples):
            self._domain_index[sample.domain].append(start_idx + i)

    def add_texts(
        self,
        texts: List[str],
        source_name: str = "custom",
        domain: str = "general",
    ):
        """Add raw texts as a new data source."""
        samples = []
        for text in texts:
            quality = self.quality_scorer.score(text)
            sample = DataSample(
                text=text,
                domain=domain,
                quality_score=quality,
                source=source_name,
            )
            samples.append(sample)
        source = DataSource(
            name=source_name,
            samples=samples,
            domain=domain,
        )
        self.add_source(source)

    def curate(
        self,
        max_samples: int = 100000,
        quality_threshold: float = 0.3,
        domain_balance: bool = True,
        difficulty_order: bool = False,
    ) -> List[DataSample]:
        """Curate a training dataset.

        Args:
            max_samples: Maximum number of samples.
            quality_threshold: Minimum quality score.
            domain_balance: Whether to balance across domains.
            difficulty_order: Whether to order by difficulty.

        Returns:
            Curated list of DataSample objects.
        """
        filtered = [
            s for s in self._all_samples
            if s.quality_score >= quality_threshold and s.is_active
        ]

        if difficulty_order:
            filtered.sort(key=lambda s: s.difficulty_score)

        if domain_balance and self._domain_index:
            balanced = self._balance_domains(filtered, max_samples)
        else:
            balanced = filtered[:max_samples]

        return balanced

    def _balance_domains(
        self, samples: List[DataSample], max_samples: int
    ) -> List[DataSample]:
        """Balance samples across domains."""
        domain_samples: Dict[str, List[DataSample]] = defaultdict(list)
        for sample in samples:
            domain_samples[sample.domain].append(sample)

        domains = list(domain_samples.keys())
        per_domain = max(1, max_samples // len(domains))

        balanced = []
        for domain in domains:
            domain_list = domain_samples[domain]
            self.rng.shuffle(domain_list)
            balanced.extend(domain_list[:per_domain])

        self.rng.shuffle(balanced)
        return balanced[:max_samples]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about curated data."""
        domain_counts = Counter(s.domain for s in self._all_samples)
        quality_scores = [s.quality_score for s in self._all_samples]

        return {
            "total_samples": len(self._all_samples),
            "total_sources": len(self._sources),
            "domains": dict(domain_counts),
            "avg_quality": sum(quality_scores) / max(1, len(quality_scores)),
            "source_stats": {name: src.to_dict() for name, src in self._sources.items()},
        }


# ---------------------------------------------------------------------------
# Domain Balancer
# ---------------------------------------------------------------------------

class DomainBalancer:
    """Balance training data across domains.

    Ensures that no single domain dominates training, while allowing
    configurable weighting and prioritization.

    Args:
        target_ratios: Target ratio for each domain.
        seed: Random seed.
    """

    def __init__(
        self,
        target_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.target_ratios = target_ratios or {}
        self.rng = random.Random(seed)
        self._domain_samples: Dict[str, List[DataSample]] = defaultdict(list)
        self._current_ratios: Dict[str, float] = {}

    def add_samples(self, samples: List[DataSample]):
        """Add samples indexed by domain."""
        for sample in samples:
            self._domain_samples[sample.domain].append(sample)

    def balance(
        self,
        total_budget: int = 100000,
        min_per_domain: int = 100,
    ) -> List[DataSample]:
        """Create a balanced dataset.

        Args:
            total_budget: Total number of samples.
            min_per_domain: Minimum samples per domain.

        Returns:
            Balanced list of samples.
        """
        domains = list(self._domain_samples.keys())

        if not self.target_ratios:
            uniform_ratio = 1.0 / max(1, len(domains))
            self.target_ratios = {d: uniform_ratio for d in domains}

        total_ratio = sum(self.target_ratios.get(d, 0) for d in domains)
        normalized = {d: r / total_ratio for d, r in self.target_ratios.items()}

        result = []
        for domain in domains:
            available = self._domain_samples[domain]
            target_count = max(min_per_domain, int(total_budget * normalized.get(domain, 0)))

            self.rng.shuffle(available)
            selected = available[:min(target_count, len(available))]
            result.extend(selected)

            self._current_ratios[domain] = len(selected) / max(1, total_budget)

        self.rng.shuffle(result)
        return result[:total_budget]

    def get_current_distribution(self) -> Dict[str, float]:
        """Get the current domain distribution."""
        return dict(self._current_ratios)

    def update_ratios(self, new_ratios: Dict[str, float]):
        """Update target domain ratios."""
        self.target_ratios = new_ratios


# ---------------------------------------------------------------------------
# Difficulty Sampler
# ---------------------------------------------------------------------------

class DifficultySampler:
    """Sample data by difficulty (easy to hard curriculum).

    Provides sampling that favors easier samples early and gradually
    introduces harder ones.

    Args:
        samples: List of data samples.
        initial_difficulty: Starting difficulty threshold.
        growth_rate: How fast difficulty increases.
        seed: Random seed.
    """

    def __init__(
        self,
        samples: Optional[List[DataSample]] = None,
        initial_difficulty: float = 0.1,
        growth_rate: float = 0.001,
        seed: int = 42,
    ):
        self.samples = samples or []
        self.initial_difficulty = initial_difficulty
        self.growth_rate = growth_rate
        self.rng = random.Random(seed)
        self._current_difficulty = initial_difficulty
        self._step = 0

        if self.samples:
            self.samples.sort(key=lambda s: s.difficulty_score)

    def set_samples(self, samples: List[DataSample]):
        """Set the sample pool."""
        self.samples = samples
        self.samples.sort(key=lambda s: s.difficulty_score)

    def update(self):
        """Update the difficulty threshold."""
        self._step += 1
        self._current_difficulty = min(
            1.0,
            self.initial_difficulty + self._step * self.growth_rate,
        )

    def sample(self, batch_size: int) -> List[DataSample]:
        """Sample a batch based on current difficulty.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            List of sampled DataSample objects.
        """
        eligible = [
            s for s in self.samples
            if s.difficulty_score <= self._current_difficulty and s.is_active
        ]

        if not eligible:
            eligible = self.samples[:batch_size] if self.samples else []

        weights = []
        for s in eligible:
            distance = abs(s.difficulty_score - self._current_difficulty)
            weight = max(0.1, 1.0 - distance * 2)
            weights.append(weight)

        total = sum(weights)
        if total > 0:
            probs = [w / total for w in weights]
        else:
            probs = [1.0 / len(eligible)] * len(eligible)

        k = min(batch_size, len(eligible))
        selected = self.rng.choices(eligible, weights=probs, k=k)
        return selected

    def get_difficulty(self) -> float:
        """Return the current difficulty threshold."""
        return self._current_difficulty


# ---------------------------------------------------------------------------
# Data Deduplication (MinHash + LSH)
# ---------------------------------------------------------------------------

class DataDeduplication:
    """MinHash + LSH based deduplication for large datasets.

    Computes MinHash signatures for documents and uses Locality-Sensitive
    Hashing to efficiently find near-duplicate pairs.

    Args:
        num_hashes: Number of hash functions for MinHash.
        num_bands: Number of bands for LSH.
        band_size: Size of each band (rows per band).
        jaccard_threshold: Jaccard similarity threshold for duplicates.
        n_grams: N-gram size for shingling.
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        band_size: Optional[int] = None,
        jaccard_threshold: float = 0.8,
        n_grams: int = 5,
    ):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.band_size = band_size or (num_hashes // num_bands)
        self.jaccard_threshold = jaccard_threshold
        self.n_grams = n_grams
        self._hash_seeds = list(range(num_hashes))
        self._signatures: Dict[str, List[int]] = {}
        self._lsh_buckets: Dict[int, Set[str]] = defaultdict(set)
        self._exact_hashes: Set[str] = set()

    def _shingle(self, text: str) -> Set[str]:
        """Create n-gram shingles from text."""
        words = text.lower().split()
        if len(words) < self.n_grams:
            return {text.lower()}
        shingles = set()
        for i in range(len(words) - self.n_grams + 1):
            shingle = " ".join(words[i:i + self.n_grams])
            shingles.add(shingle)
        return shingles

    def _exact_hash(self, text: str) -> str:
        """Compute exact hash for deduplication."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature for a set of shingles."""
        signature = []
        for seed in self._hash_seeds:
            min_hash = float("inf")
            for shingle in shingles:
                h = int(hashlib.sha256(f"{seed}:{shingle}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _compute_lsh(self, sample_id: str, signature: List[int]):
        """Compute LSH bands and add to buckets."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = min(start + self.band_size, len(signature))
            if start >= len(signature):
                break
            band = tuple(signature[start:end])
            band_hash = hash(band)
            self._lsh_buckets[band_hash].add(sample_id)

    def add_document(self, text: str, sample_id: Optional[str] = None):
        """Add a document for deduplication.

        Args:
            text: Document text.
            sample_id: Optional unique identifier.
        """
        if sample_id is None:
            sample_id = self._exact_hash(text)

        exact_h = self._exact_hash(text)
        self._exact_hashes.add(exact_h)

        shingles = self._shingle(text)
        signature = self._minhash(shingles)
        self._signatures[sample_id] = signature
        self._compute_lsh(sample_id, signature)

    def add_documents(self, texts: List[str], sample_ids: Optional[List[str]] = None):
        """Add multiple documents."""
        ids = sample_ids or [self._exact_hash(t) for t in texts]
        for text, sid in zip(texts, ids):
            self.add_document(text, sid)

    def is_duplicate(self, text: str, sample_id: Optional[str] = None) -> bool:
        """Check if a document is a duplicate.

        Args:
            text: Document text to check.
            sample_id: Optional sample identifier.

        Returns:
            True if the document is a near-duplicate of an existing one.
        """
        exact_h = self._exact_hash(text)
        if exact_h in self._exact_hashes:
            return True

        if sample_id is None:
            sample_id = exact_h

        shingles = self._shingle(text)
        signature = self._minhash(shingles)

        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = min(start + self.band_size, len(signature))
            if start >= len(signature):
                break
            band = tuple(signature[start:end])
            band_hash = hash(band)

            if band_hash in self._lsh_buckets:
                for candidate_id in self._lsh_buckets[band_hash]:
                    if candidate_id == sample_id:
                        continue
                    candidate_sig = self._signatures.get(candidate_id)
                    if candidate_sig:
                        similarity = self._jaccard_similarity(signature, candidate_sig)
                        if similarity >= self.jaccard_threshold:
                            return True

        return False

    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def deduplicate(
        self, texts: List[str]
    ) -> Tuple[List[str], List[int]]:
        """Deduplicate a list of texts.

        Args:
            texts: List of texts to deduplicate.

        Returns:
            Tuple of (unique_texts, duplicate_indices).
        """
        unique_texts = []
        duplicate_indices = []

        for i, text in enumerate(texts):
            if not self.is_duplicate(text):
                self.add_document(text)
                unique_texts.append(text)
            else:
                duplicate_indices.append(i)

        return unique_texts, duplicate_indices

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "total_documents": len(self._signatures),
            "total_lsh_buckets": len(self._lsh_buckets),
            "exact_hashes": len(self._exact_hashes),
            "num_hashes": self.num_hashes,
            "num_bands": self.num_bands,
            "band_size": self.band_size,
            "jaccard_threshold": self.jaccard_threshold,
        }


# ---------------------------------------------------------------------------
# Data Mixer
# ---------------------------------------------------------------------------

class DataMixer:
    """Mix multiple data sources with configurable ratios.

    Manages the combination of multiple datasets according to
    specified mixing ratios.

    Args:
        sources: Dictionary of source name -> list of samples.
        ratios: Dictionary of source name -> mixing ratio.
        seed: Random seed.
    """

    def __init__(
        self,
        sources: Optional[Dict[str, List[DataSample]]] = None,
        ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.sources = sources or {}
        self.ratios = ratios or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self._epoch_indices: Dict[str, List[int]] = {}
        self._epoch = 0

    def set_source(self, name: str, samples: List[DataSample]):
        """Set a data source."""
        self.sources[name] = samples
        indices = list(range(len(samples)))
        self.rng.shuffle(indices)
        self._epoch_indices[name] = indices

    def set_ratio(self, name: str, ratio: float):
        """Set mixing ratio for a source."""
        self.ratios[name] = ratio

    def set_ratios(self, ratios: Dict[str, float]):
        """Set all mixing ratios."""
        self.ratios = ratios

    def sample_batch(self, batch_size: int) -> List[DataSample]:
        """Sample a mixed batch.

        Args:
            batch_size: Number of samples.

        Returns:
            List of mixed samples.
        """
        active_sources = {
            name: samples for name, samples in self.sources.items()
            if name in self.ratios and self.ratios[name] > 0 and samples
        }

        if not active_sources:
            return []

        total_ratio = sum(self.ratios.get(name, 0) for name in active_sources)
        if total_ratio <= 0:
            return []

        batch = []
        for name, samples in active_sources.items():
            fraction = self.ratios[name] / total_ratio
            count = max(1, int(batch_size * fraction))

            indices = self._epoch_indices.get(name, list(range(len(samples))))
            selected_indices = []
            while len(selected_indices) < count and indices:
                idx = indices.pop(0)
                selected_indices.append(idx)
                indices.append(idx)

            self._epoch_indices[name] = indices

            for idx in selected_indices:
                if idx < len(samples):
                    batch.append(samples[idx])

        self.rng.shuffle(batch)
        return batch[:batch_size]

    def get_epoch_iterator(
        self, batch_size: int, num_batches: int
    ) -> Iterator[List[DataSample]]:
        """Get an iterator over batches for one epoch.

        Args:
            batch_size: Samples per batch.
            num_batches: Number of batches.

        Yields:
            Lists of mixed samples.
        """
        for _ in range(num_batches):
            yield self.sample_batch(batch_size)

    def get_effective_ratios(self) -> Dict[str, float]:
        """Get normalized effective ratios."""
        active = {k: v for k, v in self.ratios.items() if v > 0 and k in self.sources}
        total = sum(active.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in active.items()}

    def get_source_sizes(self) -> Dict[str, int]:
        """Get sizes of all sources."""
        return {name: len(samples) for name, samples in self.sources.items()}


# ---------------------------------------------------------------------------
# Token Budget Manager
# ---------------------------------------------------------------------------

class TokenBudgetManager:
    """Manage total token budget across training.

    Tracks token consumption per source and enforces budget limits.

    Args:
        total_budget: Total token budget for training.
    """

    def __init__(self, total_budget: int = 1_000_000_000):
        self.total_budget = total_budget
        self._used_tokens = 0
        self._source_tokens: Counter = Counter()
        self._allocations: Dict[str, BudgetAllocation] = {}
        self._epoch_tokens: int = 0
        self._step_tokens: int = 0

    def allocate(self, source_name: str, tokens: int):
        """Allocate a token budget for a source."""
        available = self.total_budget - self._used_tokens
        allocated = min(tokens, available)
        self._allocations[source_name] = BudgetAllocation(
            source_name=source_name,
            allocated_tokens=allocated,
        )
        logger.info(f"Budget: allocated {allocated} tokens to '{source_name}'")

    def consume(self, source_name: str, tokens: int) -> bool:
        """Consume tokens from a source's allocation.

        Args:
            source_name: Name of the data source.
            tokens: Number of tokens consumed.

        Returns:
            True if tokens were consumed, False if budget exceeded.
        """
        if self._used_tokens + tokens > self.total_budget:
            return False

        self._used_tokens += tokens
        self._source_tokens[source_name] += tokens
        self._step_tokens += tokens
        self._epoch_tokens += tokens

        if source_name in self._allocations:
            self._allocations[source_name].consume(tokens)

        return True

    def record_batch(self, tokens: int, source: str = "training"):
        """Record a batch of tokens."""
        self.consume(source, tokens)

    def start_epoch(self):
        """Mark the start of a new epoch."""
        self._epoch_tokens = 0

    def start_step(self):
        """Mark the start of a new step."""
        self._step_tokens = 0

    def get_remaining_budget(self) -> int:
        """Return remaining token budget."""
        return max(0, self.total_budget - self._used_tokens)

    def get_utilization(self) -> float:
        """Return budget utilization fraction."""
        return self._used_tokens / max(1, self.total_budget)

    def get_source_breakdown(self) -> Dict[str, int]:
        """Return token consumption per source."""
        return dict(self._source_tokens)

    def get_report(self) -> Dict[str, Any]:
        """Get a budget report."""
        return {
            "total_budget": self.total_budget,
            "used_tokens": self._used_tokens,
            "remaining_tokens": self.get_remaining_budget(),
            "utilization": self.get_utilization(),
            "source_breakdown": self.get_source_breakdown(),
            "allocations": {
                name: {"allocated": a.allocated_tokens, "used": a.used_tokens,
                         "utilization": a.utilization}
                for name, a in self._allocations.items()
            },
        }

    def estimate_epochs_remaining(
        self, tokens_per_epoch: int
    ) -> float:
        """Estimate remaining epochs given tokens per epoch."""
        remaining = self.get_remaining_budget()
        return remaining / max(1, tokens_per_epoch)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Store and replay important examples during training.

    Maintains a buffer of high-value training examples that are
    periodically replayed to prevent catastrophic forgetting.

    Args:
        capacity: Maximum number of examples in the buffer.
        selection_criteria: How to select examples for replay.
        seed: Random seed.
    """

    def __init__(
        self,
        capacity: int = 10000,
        selection_criteria: str = "quality",
        seed: int = 42,
    ):
        self.capacity = capacity
        self.selection_criteria = selection_criteria
        self.rng = random.Random(seed)
        self._buffer: List[DataSample] = []
        self._priorities: List[float] = []
        self._sample_counts: Counter = Counter()

    def add(self, sample: DataSample, priority: Optional[float] = None):
        """Add a sample to the buffer.

        Args:
            sample: DataSample to add.
            priority: Optional priority value.
        """
        if priority is None:
            if self.selection_criteria == "quality":
                priority = sample.quality_score
            elif self.selection_criteria == "difficulty":
                priority = sample.difficulty_score
            else:
                priority = 1.0

        if len(self._buffer) >= self.capacity:
            min_idx = min(range(len(self._priorities)), key=lambda i: self._priorities[i])
            if priority > self._priorities[min_idx]:
                self._buffer[min_idx] = sample
                self._priorities[min_idx] = priority
        else:
            self._buffer.append(sample)
            self._priorities.append(priority)

    def add_batch(self, samples: List[DataSample]):
        """Add multiple samples."""
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int) -> List[DataSample]:
        """Sample from the replay buffer.

        Args:
            batch_size: Number of samples.

        Returns:
            List of replayed samples.
        """
        if not self._buffer:
            return []

        weights = []
        for i, sample in enumerate(self._buffer):
            freq_penalty = 1.0 / (1.0 + self._sample_counts[sample.sample_id])
            weight = self._priorities[i] * freq_penalty
            weights.append(max(0.01, weight))

        total = sum(weights)
        probs = [w / total for w in weights]

        k = min(batch_size, len(self._buffer))
        selected = self.rng.choices(self._buffer, weights=probs, k=k)

        for s in selected:
            self._sample_counts[s.sample_id] += 1

        return selected

    def update_priority(self, sample_id: str, new_priority: float):
        """Update priority of a sample."""
        for i, sample in enumerate(self._buffer):
            if sample.sample_id == sample_id:
                self._priorities[i] = new_priority
                break

    def get_buffer_size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        qualities = [s.quality_score for s in self._buffer]
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "utilization": len(self._buffer) / max(1, self.capacity),
            "avg_quality": sum(qualities) / max(1, len(qualities)),
            "total_replays": sum(self._sample_counts.values()),
            "unique_replayed": len(self._sample_counts),
        }


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """End-to-end data processing pipeline.

    Chains together data loading, filtering, scoring, deduplication,
    mixing, and batching into a unified pipeline.

    Args:
        config: Pipeline configuration dictionary.
        seed: Random seed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.quality_scorer = QualityScorer()
        self.deduplicator = DataDeduplication()
        self.domain_balancer = DomainBalancer(seed=seed)
        self.difficulty_sampler = DifficultySampler(seed=seed)
        self.mixer = DataMixer(seed=seed)
        self.budget_manager = TokenBudgetManager(
            self.config.get("total_budget", 1_000_000_000)
        )
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.get("replay_capacity", 10000),
            seed=seed,
        )
        self._samples: List[DataSample] = []
        self._is_built = False

    def load_texts(
        self,
        texts: List[str],
        source_name: str = "main",
        domain: str = "general",
        compute_quality: bool = True,
    ):
        """Load texts into the pipeline.

        Args:
            texts: List of text strings.
            source_name: Source identifier.
            domain: Data domain.
            compute_quality: Whether to compute quality scores.
        """
        samples = []
        for text in texts:
            quality = self.quality_scorer.score(text) if compute_quality else 0.5
            sample = DataSample(
                text=text,
                domain=domain,
                quality_score=quality,
                source=source_name,
            )
            samples.append(sample)

        self._samples.extend(samples)
        self.domain_balancer.add_samples(samples)
        logger.info(f"Loaded {len(samples)} samples from '{source_name}' ({domain})")

    def filter_by_quality(self, min_quality: float = 0.3) -> int:
        """Filter samples by quality score.

        Args:
            min_quality: Minimum quality threshold.

        Returns:
            Number of filtered samples.
        """
        before = len(self._samples)
        self._samples = [s for s in self._samples if s.quality_score >= min_quality]
        removed = before - len(self._samples)
        logger.info(f"Quality filter: removed {removed}, {len(self._samples)} remaining")
        return removed

    def deduplicate(self, threshold: float = 0.8) -> int:
        """Remove duplicate samples.

        Args:
            threshold: Jaccard similarity threshold.

        Returns:
            Number of duplicates removed.
        """
        self.deduplicator.jaccard_threshold = threshold
        before = len(self._samples)
        unique_texts = []
        seen_ids = set()

        for sample in self._samples:
            if not self.deduplicator.is_duplicate(sample.text, sample.sample_id):
                self.deduplicator.add_document(sample.text, sample.sample_id)
                unique_texts.append(sample)
            else:
                seen_ids.add(sample.sample_id)

        removed = before - len(unique_texts)
        self._samples = unique_texts
        logger.info(f"Deduplication: removed {removed} duplicates")
        return removed

    def balance_domains(
        self,
        max_samples: int = 100000,
        min_per_domain: int = 100,
    ) -> List[DataSample]:
        """Balance domain distribution.

        Args:
            max_samples: Maximum total samples.
            min_per_domain: Minimum per domain.

        Returns:
            Balanced list of samples.
        """
        balanced = self.domain_balancer.balance(max_samples, min_per_domain)
        self._samples = balanced
        return balanced

    def build(self) -> List[DataSample]:
        """Build the final dataset.

        Returns:
            Processed list of samples.
        """
        self._is_built = True
        logger.info(f"Pipeline built: {len(self._samples)} samples")
        return self._samples

    def get_batch(self, batch_size: int) -> List[DataSample]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples.

        Returns:
            List of samples.
        """
        if not self._samples:
            return []

        self.difficulty_sampler.update()
        curriculum_batch = self.difficulty_sampler.sample(batch_size // 2)

        replay_batch = []
        if self.replay_buffer.get_buffer_size() > 0:
            replay_batch = self.replay_buffer.sample(batch_size // 4)

        remaining = batch_size - len(curriculum_batch) - len(replay_batch)
        random_batch = self.rng.sample(
            self._samples, min(remaining, len(self._samples))
        )

        batch = curriculum_batch + replay_batch + random_batch
        self.rng.shuffle(batch)

        total_tokens = sum(s.token_count for s in batch)
        self.budget_manager.record_batch(total_tokens)

        return batch[:batch_size]

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        domain_counts = Counter(s.domain for s in self._samples)
        return {
            "total_samples": len(self._samples),
            "is_built": self._is_built,
            "domain_distribution": dict(domain_counts),
            "budget": self.budget_manager.get_report(),
            "replay_buffer": self.replay_buffer.get_stats(),
            "deduplication": self.deduplicator.get_stats(),
        }


# ---------------------------------------------------------------------------
# Streaming Data Loader
# ---------------------------------------------------------------------------

class StreamingDataLoader:
    """Load data in streaming fashion from disk.

    Efficiently loads and processes data without loading everything
    into memory, supporting JSONL and JSON formats.

    Args:
        file_path: Path to data file.
        batch_size: Batch size.
        text_field: Field name containing text.
        shuffle: Whether to shuffle.
        buffer_size: Shuffle buffer size.
        max_samples: Maximum samples to load (-1 for all).
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int = 32,
        text_field: str = "text",
        shuffle: bool = True,
        buffer_size: int = 10000,
        max_samples: int = -1,
    ):
        self.file_path = file_path
        self.batch_size = batch_size
        self.text_field = text_field
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.max_samples = max_samples
        self._buffer: deque = deque(maxlen=buffer_size)
        self._file_handle = None
        self._total_read = 0
        self._is_exhausted = False

    def open(self):
        """Open the data file for streaming."""
        if self.file_path.endswith(".jsonl"):
            self._file_handle = open(self.file_path, "r", encoding="utf-8")
        elif self.file_path.endswith(".json"):
            self._file_handle = open(self.file_path, "r", encoding="utf-8")
        else:
            self._file_handle = open(self.file_path, "r", encoding="utf-8")
        self._fill_buffer()
        self._is_exhausted = False
        self._total_read = 0
        logger.info(f"StreamingDataLoader: opened {self.file_path}")

    def close(self):
        """Close the data file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self._buffer.clear()

    def _fill_buffer(self):
        """Fill the internal buffer from the file."""
        if self._is_exhausted or not self._file_handle:
            return

        while len(self._buffer) < self.buffer_size:
            line = self._file_handle.readline()
            if not line:
                self._is_exhausted = True
                break

            line = line.strip()
            if not line:
                continue

            try:
                if self.file_path.endswith(".jsonl"):
                    data = json.loads(line)
                elif self.file_path.endswith(".json"):
                    continue
                else:
                    data = {self.text_field: line}

                text = data.get(self.text_field, "")
                if text:
                    self._buffer.append(data)
                    self._total_read += 1

                    if 0 < self.max_samples <= self._total_read:
                        self._is_exhausted = True
                        break

            except json.JSONDecodeError:
                continue

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over batches."""
        if self._file_handle is None:
            self.open()

        while True:
            if len(self._buffer) < self.batch_size:
                self._fill_buffer()

            if not self._buffer:
                break

            batch = []
            for _ in range(min(self.batch_size, len(self._buffer))):
                if self._buffer:
                    batch.append(self._buffer.popleft())

            if self.shuffle:
                random.shuffle(batch)

            if batch:
                yield batch

    def __len__(self) -> int:
        """Return total samples read so far."""
        return self._total_read

    def has_more(self) -> bool:
        """Check if there are more samples."""
        return bool(self._buffer) or not self._is_exhausted

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        file_size = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        return {
            "file_path": self.file_path,
            "file_size_mb": file_size / (1024 * 1024),
            "total_read": self._total_read,
            "buffer_size": len(self._buffer),
            "max_buffer": self.buffer_size,
            "is_exhausted": self._is_exhausted,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_token_estimate(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text.

    Args:
        text: Input text.
        chars_per_token: Average characters per token.

    Returns:
        Estimated token count.
    """
    return max(1, int(len(text) / chars_per_token))


def analyze_dataset(
    texts: List[str],
    scorer: Optional[QualityScorer] = None,
) -> Dict[str, Any]:
    """Analyze a dataset and return statistics.

    Args:
        texts: List of texts.
        scorer: Optional quality scorer.

    Returns:
        Dictionary of dataset statistics.
    """
    if not texts:
        return {"num_texts": 0}

    lengths = [len(t.split()) for t in texts]
    char_lengths = [len(t) for t in texts]
    token_estimates = [compute_token_estimate(t) for t in texts]

    stats = {
        "num_texts": len(texts),
        "avg_word_length": sum(lengths) / len(lengths),
        "median_word_length": sorted(lengths)[len(lengths) // 2],
        "min_word_length": min(lengths),
        "max_word_length": max(lengths),
        "avg_char_length": sum(char_lengths) / len(char_lengths),
        "total_tokens_estimate": sum(token_estimates),
        "avg_tokens_estimate": sum(token_estimates) / len(token_estimates),
    }

    if scorer:
        quality_scores = [scorer.score(t) for t in texts]
        stats["avg_quality"] = sum(quality_scores) / len(quality_scores)
        stats["min_quality"] = min(quality_scores)
        stats["max_quality"] = max(quality_scores)

    return stats


def create_balanced_dataset(
    texts_by_domain: Dict[str, List[str]],
    max_per_domain: int = 10000,
    quality_threshold: float = 0.3,
    seed: int = 42,
) -> List[DataSample]:
    """Create a domain-balanced dataset from text collections.

    Args:
        texts_by_domain: Dictionary mapping domain to texts.
        max_per_domain: Maximum samples per domain.
        quality_threshold: Minimum quality score.
        seed: Random seed.

    Returns:
        List of balanced DataSample objects.
    """
    scorer = QualityScorer()
    rng = random.Random(seed)
    result = []

    for domain, texts in texts_by_domain.items():
        samples = []
        for text in texts:
            quality = scorer.score(text)
            if quality >= quality_threshold:
                samples.append(DataSample(
                    text=text,
                    domain=domain,
                    quality_score=quality,
                ))
        rng.shuffle(samples)
        result.extend(samples[:max_per_domain])

    rng.shuffle(result)
    return result


def export_data_report(
    pipeline: DataPipeline,
    path: str,
):
    """Export a data pipeline report to JSON.

    Args:
        pipeline: DataPipeline instance.
        path: Output file path.
    """
    stats = pipeline.get_stats()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Data report exported to {path}")
