"""
Data Visualizer - Dataset Analysis
====================================

Comprehensive tools for analyzing LLM training datasets including
token statistics, quality analysis, duplicate detection, and
class balance checking.

All implementations use Python stdlib only.
"""

import math
import json
import os
import hashlib
import time
import re
import string
import threading
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Sequence, Set, Iterator
)
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_H = "─"
BOX_V = "│"
BOX_LT = "├"
BOX_RT = "┤"
BOX_BT = "┬"
BOX_BB = "┴"
BOX_CROSS = "┼"

PROGRESS_FULL = "█"
PROGRESS_THREE_QUARTERS = "▓"
PROGRESS_HALF = "▒"
PROGRESS_QUARTER = "░"
PROGRESS_EMPTY = " "


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TokenStats:
    """Token-level statistics for a dataset."""
    total_tokens: int = 0
    total_documents: int = 0
    avg_tokens_per_doc: float = 0.0
    median_tokens_per_doc: float = 0.0
    min_tokens_per_doc: int = 0
    max_tokens_per_doc: int = 0
    std_tokens_per_doc: float = 0.0
    percentile_25: int = 0
    percentile_75: int = 0
    percentile_90: int = 0
    percentile_95: int = 0
    percentile_99: int = 0
    token_length_histogram: Dict[int, int] = field(default_factory=dict)
    empty_documents: int = 0
    very_short_documents: int = 0  # < 10 tokens
    very_long_documents: int = 0   # > 10000 tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "total_documents": self.total_documents,
            "avg_tokens_per_doc": self.avg_tokens_per_doc,
            "median_tokens_per_doc": self.median_tokens_per_doc,
            "min_tokens_per_doc": self.min_tokens_per_doc,
            "max_tokens_per_doc": self.max_tokens_per_doc,
            "std_tokens_per_doc": self.std_tokens_per_doc,
            "percentile_25": self.percentile_25,
            "percentile_75": self.percentile_75,
            "percentile_90": self.percentile_90,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "empty_documents": self.empty_documents,
            "very_short_documents": self.very_short_documents,
            "very_long_documents": self.very_long_documents,
        }


@dataclass
class VocabStats:
    """Vocabulary statistics."""
    vocab_size: int = 0
    unique_tokens: int = 0
    coverage_ratio: float = 0.0
    top_tokens: List[Tuple[str, int]] = field(default_factory=list)
    rare_tokens: int = 0
    oov_estimate: float = 0.0
    token_frequency_distribution: Dict[str, int] = field(default_factory=dict)
    type_token_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "unique_tokens": self.unique_tokens,
            "coverage_ratio": self.coverage_ratio,
            "top_tokens": self.top_tokens[:20],
            "rare_tokens": self.rare_tokens,
            "oov_estimate": self.oov_estimate,
            "type_token_ratio": self.type_token_ratio,
        }


@dataclass
class QualityIssue:
    """A detected data quality issue."""
    issue_type: str = ""
    severity: str = "info"  # info, warning, error
    description: str = ""
    count: int = 0
    examples: List[str] = field(default_factory=list)
    affected_indices: List[int] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    total_documents: int = 0
    total_issues: int = 0
    critical_issues: int = 0
    warnings: int = 0
    info_count: int = 0
    duplicate_count: int = 0
    near_duplicate_count: int = 0
    anomaly_count: int = 0
    empty_count: int = 0
    issues: List[QualityIssue] = field(default_factory=list)
    overall_score: float = 1.0  # 0-1, 1 = perfect

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "duplicate_count": self.duplicate_count,
            "near_duplicate_count": self.near_duplicate_count,
            "anomaly_count": self.anomaly_count,
            "empty_count": self.empty_count,
            "overall_score": self.overall_score,
            "issues": [
                {"type": i.issue_type, "severity": i.severity,
                 "description": i.description, "count": i.count}
                for i in self.issues
            ],
        }


@dataclass
class ClassBalanceInfo:
    """Information about class/category balance."""
    label: str = ""
    count: int = 0
    proportion: float = 0.0
    ideal_proportion: float = 0.0
    imbalance_ratio: float = 1.0
    sampling_weight: float = 1.0


@dataclass
class DataProfile:
    """Complete dataset profile."""
    name: str = "dataset"
    token_stats: TokenStats = field(default_factory=TokenStats)
    vocab_stats: VocabStats = field(default_factory=VocabStats)
    quality_report: DataQualityReport = field(default_factory=DataQualityReport)
    language_distribution: Dict[str, float] = field(default_factory=dict)
    profile_time: float = 0.0

    def summary(self) -> str:
        """Get text summary."""
        lines = [
            f"Dataset Profile: {self.name}",
            f"  Documents:      {self.token_stats.total_documents:>15,}",
            f"  Total tokens:   {self.token_stats.total_tokens:>15,}",
            f"  Avg tokens/doc: {self.token_stats.avg_tokens_per_doc:>15.1f}",
            f"  Unique tokens:  {self.vocab_stats.unique_tokens:>15,}",
            f"  Quality score:  {self.quality_report.overall_score:>15.2%}",
            f"  Languages:      {len(self.language_distribution):>15}",
        ]
        return "\n".join(lines)


# ============================================================================
# Helper Functions
# ============================================================================

def _simple_tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer.

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    if not text:
        return []
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from text.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        List of character n-grams.
    """
    if not text or n <= 0:
        return []
    text = text.lower().strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def _word_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract word n-grams from text.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        List of word n-grams.
    """
    tokens = _simple_tokenize(text)
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text.

    Args:
        text: Input text.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Jaccard similarity (0-1).
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _cosine_similarity_counts(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Compute cosine similarity between two count dictionaries.

    Args:
        counts_a: First count dictionary.
        counts_b: Second count dictionary.

    Returns:
        Cosine similarity (0-1).
    """
    all_keys = set(counts_a.keys()) | set(counts_b.keys())
    if not all_keys:
        return 1.0

    dot = sum(counts_a.get(k, 0) * counts_b.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counts_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_percentile(sorted_values: List[int], percentile: float) -> int:
    """Compute percentile from sorted values.

    Args:
        sorted_values: Sorted list of integer values.
        percentile: Percentile (0-100).

    Returns:
        Value at the given percentile.
    """
    if not sorted_values:
        return 0
    idx = int(len(sorted_values) * percentile / 100)
    idx = max(0, min(len(sorted_values) - 1, idx))
    return sorted_values[idx]


def _detect_language_simple(text: str) -> str:
    """Simple language detection based on character ranges.

    Args:
        text: Input text.

    Returns:
        Language code string.
    """
    if not text:
        return "unknown"

    text = text[:500]  # Sample first 500 chars
    latin = sum(1 for c in text if c.isalpha() and c in string.ascii_letters)
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    cjk = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF' or '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF')
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    thai = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')

    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "unknown"

    ratios = {
        "cjk": cjk / total_alpha,
        "cyrillic": cyrillic / total_alpha,
        "arabic": arabic / total_alpha,
        "devanagari": devanagari / total_alpha,
        "thai": thai / total_alpha,
        "latin": latin / total_alpha,
    }

    best_lang = max(ratios, key=ratios.get)
    if ratios[best_lang] < 0.3:
        return "mixed"
    return best_lang


# ============================================================================
# DataProfiler
# ============================================================================

class DataProfiler:
    """
    Analyze dataset statistics including token counts, sequence lengths,
    and vocabulary distribution.

    Supports both text datasets and pre-tokenized datasets.

    Example:
        profiler = DataProfiler()
        profile = profiler.profile(texts)
        print(profile.summary())
        print(profiler.token_length_distribution_chart())
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize the data profiler.

        Args:
            tokenizer: Optional tokenizer function (text -> list of tokens).
            max_samples: Maximum samples to profile (None for all).
        """
        self._tokenizer = tokenizer or _simple_tokenize
        self._max_samples = max_samples

    def profile(
        self,
        dataset: Any,
        num_samples: Optional[int] = None,
        name: str = "dataset",
    ) -> DataProfile:
        """Profile a dataset.

        Args:
            dataset: Dataset to profile. Can be a list of strings, a list of
                    token lists, or an object with __iter__ or __getitem__.
            num_samples: Number of samples to profile.
            name: Dataset name.

        Returns:
            DataProfile with complete analysis.
        """
        start_time = time.time()
        profile = DataProfile(name=name)

        # Extract texts
        texts = self._extract_texts(dataset)
        n = num_samples or self._max_samples or len(texts)
        texts = texts[:n]

        # Token statistics
        token_lengths = []
        all_tokens = []
        token_counter = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            token_lengths.append(len(tokens))
            all_tokens.extend(tokens)
            token_counter.update(tokens)

        profile.token_stats = self._compute_token_stats(token_lengths, all_tokens)
        profile.vocab_stats = self._compute_vocab_stats(token_counter, all_tokens)

        # Language distribution
        profile.language_distribution = self._compute_language_distribution(texts)

        profile.profile_time = time.time() - start_time
        return profile

    def token_length_distribution(
        self,
        dataset: Any,
        num_samples: Optional[int] = None,
    ) -> Dict[int, int]:
        """Compute token length distribution.

        Args:
            dataset: Dataset.
            num_samples: Number of samples.

        Returns:
            Dictionary mapping length -> count.
        """
        texts = self._extract_texts(dataset)
        n = num_samples or self._max_samples or len(texts)
        texts = texts[:n]

        distribution = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            distribution[len(tokens)] += 1

        return dict(sorted(distribution.items()))

    def vocabulary_coverage(
        self,
        dataset: Any,
        vocab: Optional[Set[str]] = None,
        num_samples: Optional[int] = None,
    ) -> VocabStats:
        """Analyze vocabulary coverage of the dataset.

        Args:
            dataset: Dataset.
            vocab: Known vocabulary set. None to infer from dataset.
            num_samples: Number of samples.

        Returns:
            VocabStats with coverage information.
        """
        texts = self._extract_texts(dataset)
        n = num_samples or self._max_samples or len(texts)
        texts = texts[:n]

        token_counter = Counter()
        all_tokens = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            token_counter.update(tokens)

        return self._compute_vocab_stats(token_counter, all_tokens, known_vocab=vocab)

    def token_length_distribution_chart(
        self,
        distribution: Optional[Dict[int, int]] = None,
        dataset: Optional[Any] = None,
        width: int = 60,
        height: int = 12,
    ) -> str:
        """Render token length distribution as a text chart.

        Args:
            distribution: Pre-computed distribution.
            dataset: Dataset to analyze.
            width: Chart width.
            height: Chart height.

        Returns:
            Multi-line chart string.
        """
        if distribution is None:
            if dataset is not None:
                distribution = self.token_length_distribution(dataset)
            else:
                return "(no data)"

        if not distribution:
            return "(empty distribution)"

        lengths = sorted(distribution.keys())
        counts = [distribution[l] for l in lengths]
        max_count = max(counts) if counts else 0
        min_len = lengths[0]
        max_len = lengths[-1]

        if max_count == 0:
            return "(all zero counts)"

        lines = []
        lines.append(f"Token Length Distribution (range: {min_len} - {max_len})")

        # Downsample to fit width
        if len(lengths) > width:
            step = len(lengths) / width
            new_lengths = []
            new_counts = []
            for i in range(width):
                idx = int(i * step)
                new_lengths.append(lengths[idx])
                new_counts.append(counts[idx])
            lengths = new_lengths
            counts = new_counts

        # Build chart
        for row in range(height):
            threshold = max_count * (height - row) / height
            line = ""
            for count in counts:
                if count >= threshold:
                    # Choose character based on relative height
                    rel = count / max_count
                    if rel > 0.8:
                        line += PROGRESS_FULL
                    elif rel > 0.6:
                        line += PROGRESS_THREE_QUARTERS
                    elif rel > 0.4:
                        line += PROGRESS_HALF
                    elif rel > 0.2:
                        line += PROGRESS_QUARTER
                    else:
                        line += "░"
                else:
                    line += " "
            lines.append(f"  {line}")

        # X-axis labels
        x_min = f"{min_len}"
        x_max = f"{max_len}"
        lines.append(f"  {x_min}{' ' * (width - len(x_min) - len(x_max) - 1)}{x_max}")

        # Statistics
        if len(counts) > 0:
            avg_len = sum(l * c for l, c in zip(lengths, counts)) / sum(counts)
            lines.append(f"  Mean: {avg_len:.1f} | Total docs: {sum(counts):,}")

        return "\n".join(lines)

    def _extract_texts(self, dataset: Any) -> List[str]:
        """Extract text strings from a dataset.

        Args:
            dataset: Dataset object.

        Returns:
            List of text strings.
        """
        texts = []

        if isinstance(dataset, (list, tuple)):
            for item in dataset:
                texts.append(self._to_string(item))
        elif hasattr(dataset, "__iter__"):
            try:
                for item in dataset:
                    texts.append(self._to_string(item))
                    if self._max_samples and len(texts) >= self._max_samples:
                        break
            except (TypeError, StopIteration):
                pass
        elif hasattr(dataset, "__getitem__"):
            try:
                length = len(dataset)
                for i in range(min(length, self._max_samples or length)):
                    texts.append(self._to_string(dataset[i]))
            except (TypeError, IndexError):
                pass

        return texts

    def _to_string(self, item: Any) -> str:
        """Convert an item to a string.

        Args:
            item: Item to convert.

        Returns:
            String representation.
        """
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            return " ".join(str(x) for x in item)
        if isinstance(item, dict):
            if "text" in item:
                return str(item["text"])
            if "content" in item:
                return str(item["content"])
            if "prompt" in item:
                return str(item["prompt"])
            if "input" in item:
                return str(item["input"])
            return json.dumps(item, ensure_ascii=False)
        return str(item)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a text string.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        return self._tokenizer(text)

    def _compute_token_stats(
        self, lengths: List[int], all_tokens: List[str]
    ) -> TokenStats:
        """Compute token statistics.

        Args:
            lengths: Token counts per document.
            all_tokens: All tokens flattened.

        Returns:
            TokenStats.
        """
        stats = TokenStats()
        stats.total_documents = len(lengths)
        stats.total_tokens = len(all_tokens)

        if not lengths:
            return stats

        sorted_lengths = sorted(lengths)

        stats.avg_tokens_per_doc = sum(lengths) / len(lengths)
        stats.median_tokens_per_doc = sorted_lengths[len(sorted_lengths) // 2]
        stats.min_tokens_per_doc = sorted_lengths[0]
        stats.max_tokens_per_doc = sorted_lengths[-1]

        if len(lengths) > 1:
            mean = stats.avg_tokens_per_doc
            variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
            stats.std_tokens_per_doc = math.sqrt(variance)

        stats.percentile_25 = _compute_percentile(sorted_lengths, 25)
        stats.percentile_75 = _compute_percentile(sorted_lengths, 75)
        stats.percentile_90 = _compute_percentile(sorted_lengths, 90)
        stats.percentile_95 = _compute_percentile(sorted_lengths, 95)
        stats.percentile_99 = _compute_percentile(sorted_lengths, 99)

        stats.empty_documents = sum(1 for l in lengths if l == 0)
        stats.very_short_documents = sum(1 for l in lengths if 0 < l < 10)
        stats.very_long_documents = sum(1 for l in lengths if l > 10000)

        # Histogram
        hist = Counter()
        for l in lengths:
            # Bin to nearest 10
            bin_key = (l // 10) * 10
            hist[bin_key] += 1
        stats.token_length_histogram = dict(sorted(hist.items()))

        return stats

    def _compute_vocab_stats(
        self,
        token_counter: Counter,
        all_tokens: List[str],
        known_vocab: Optional[Set[str]] = None,
    ) -> VocabStats:
        """Compute vocabulary statistics.

        Args:
            token_counter: Token frequency counter.
            all_tokens: All tokens flattened.
            known_vocab: Known vocabulary set.

        Returns:
            VocabStats.
        """
        stats = VocabStats()
        stats.unique_tokens = len(token_counter)
        stats.vocab_size = known_vocab if known_vocab is not None else stats.unique_tokens

        if isinstance(stats.vocab_size, set):
            stats.vocab_size = len(stats.vocab_size)

        # Top tokens
        stats.top_tokens = token_counter.most_common(50)

        # Coverage
        if known_vocab and isinstance(known_vocab, set):
            in_vocab = sum(count for token, count in token_counter.items() if token in known_vocab)
            total = sum(token_counter.values())
            stats.coverage_ratio = in_vocab / total if total > 0 else 0.0
            stats.oov_estimate = 1.0 - stats.coverage_ratio
        else:
            stats.coverage_ratio = 1.0

        # Rare tokens (appearing only once)
        stats.rare_tokens = sum(1 for count in token_counter.values() if count == 1)

        # Type-token ratio
        total_tokens = len(all_tokens)
        stats.type_token_ratio = stats.unique_tokens / total_tokens if total_tokens > 0 else 0.0

        return stats

    def _compute_language_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Compute language distribution across texts.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary mapping language -> proportion.
        """
        lang_counts = Counter()
        for text in texts[:1000]:  # Sample first 1000
            lang = _detect_language_simple(text)
            lang_counts[lang] += 1

        total = sum(lang_counts.values())
        if total == 0:
            return {}

        return {lang: count / total for lang, count in lang_counts.most_common()}


# ============================================================================
# QualityAnalyzer
# ============================================================================

class QualityAnalyzer:
    """
    Detect data quality issues including duplicates, near-duplicates,
    anomalies, and language distribution.

    Example:
        analyzer = QualityAnalyzer()
        report = analyzer.analyze(texts)
        print(report.summary())
    """

    def __init__(
        self,
        duplicate_threshold: float = 0.95,
        near_duplicate_threshold: float = 0.8,
        anomaly_percentile: float = 99.0,
        min_text_length: int = 1,
        max_text_length: int = 1000000,
    ):
        """Initialize the quality analyzer.

        Args:
            duplicate_threshold: Jaccard similarity threshold for exact duplicates.
            near_duplicate_threshold: Threshold for near-duplicates.
            anomaly_percentile: Percentile above which texts are anomalous.
            min_text_length: Minimum acceptable text length.
            max_text_length: Maximum acceptable text length.
        """
        self._dup_thresh = duplicate_threshold
        self._near_dup_thresh = near_duplicate_threshold
        self._anomaly_pct = anomaly_percentile
        self._min_len = min_text_length
        self._max_len = max_text_length

    def analyze(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
    ) -> DataQualityReport:
        """Run full quality analysis on a list of texts.

        Args:
            texts: List of text strings.
            labels: Optional labels per text.

        Returns:
            DataQualityReport with all issues found.
        """
        report = DataQualityReport(total_documents=len(texts))

        if not texts:
            return report

        # Exact duplicates
        dup_result = self.find_duplicates(texts)
        report.duplicate_count = len(dup_result["duplicates"])

        if report.duplicate_count > 0:
            report.issues.append(QualityIssue(
                issue_type="exact_duplicates",
                severity="warning",
                description=f"Found {report.duplicate_count} exact duplicate documents",
                count=report.duplicate_count,
                examples=dup_result["examples"][:5],
            ))

        # Near duplicates
        near_dup_result = self.find_near_duplicates(texts, self._near_dup_thresh)
        report.near_duplicate_count = len(near_dup_result["pairs"])

        if report.near_duplicate_count > 0:
            report.issues.append(QualityIssue(
                issue_type="near_duplicates",
                severity="info",
                description=f"Found {report.near_duplicate_count} near-duplicate pairs (threshold={self._near_dup_thresh})",
                count=report.near_duplicate_count,
                examples=near_dup_result["examples"][:5],
            ))

        # Anomalies
        anomalies = self.detect_anomalies(texts)
        report.anomaly_count = len(anomalies)

        if report.anomaly_count > 0:
            report.issues.append(QualityIssue(
                issue_type="length_anomalies",
                severity="warning",
                description=f"Found {report.anomaly_count} documents with anomalous length",
                count=report.anomaly_count,
                examples=anomalies[:5],
            ))

        # Empty/short texts
        empty_count = sum(1 for t in texts if not t or not t.strip())
        report.empty_count = empty_count

        if empty_count > 0:
            report.issues.append(QualityIssue(
                issue_type="empty_documents",
                severity="error",
                description=f"Found {empty_count} empty documents",
                count=empty_count,
            ))

        short_count = sum(1 for t in texts if t and len(t.strip()) < self._min_len)
        if short_count > 0:
            report.issues.append(QualityIssue(
                issue_type="too_short",
                severity="warning",
                description=f"Found {short_count} documents below minimum length ({self._min_len})",
                count=short_count,
            ))

        long_count = sum(1 for t in texts if t and len(t) > self._max_len)
        if long_count > 0:
            report.issues.append(QualityIssue(
                issue_type="too_long",
                severity="info",
                description=f"Found {long_count} documents above maximum length ({self._max_len})",
                count=long_count,
            ))

        # Language distribution
        lang_dist = self.language_distribution(texts)
        if len(lang_dist) > 1:
            minority_langs = {k: v for k, v in lang_dist.items() if v < 0.01}
            if minority_langs:
                report.issues.append(QualityIssue(
                    issue_type="language_minority",
                    severity="info",
                    description=f"Found {len(minority_langs)} minority language groups (< 1%)",
                    count=sum(int(v * len(texts)) for v in minority_langs.values()),
                ))

        # Label consistency check
        if labels:
            label_issues = self._check_label_consistency(texts, labels)
            report.issues.extend(label_issues)

        # Compute overall score
        report.critical_issues = sum(1 for i in report.issues if i.severity == "error")
        report.warnings = sum(1 for i in report.issues if i.severity == "warning")
        report.info_count = sum(1 for i in report.issues if i.severity == "info")
        report.total_issues = sum(
            i.count for i in report.issues if i.severity in ("error", "warning")
        )

        # Score: penalize for issues
        penalty = 0
        penalty += report.duplicate_count * 0.01
        penalty += report.near_duplicate_count * 0.005
        penalty += report.anomaly_count * 0.01
        penalty += report.empty_count * 0.1
        penalty += report.critical_issues * 0.05
        report.overall_score = max(0, min(1, 1.0 - penalty / max(len(texts), 1)))

        return report

    def find_duplicates(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Find exact and near-exact duplicate texts.

        Args:
            texts: List of text strings.
            threshold: Override similarity threshold.

        Returns:
            Dictionary with duplicates info.
        """
        threshold = threshold or self._dup_thresh
        seen_hashes: Dict[str, List[int]] = {}
        duplicates = []
        duplicate_indices = set()

        for idx, text in enumerate(texts):
            text_hash = _hash_text(text.strip().lower())
            if text_hash in seen_hashes:
                for orig_idx in seen_hashes[text_hash]:
                    duplicates.append((orig_idx, idx, 1.0))
                duplicate_indices.add(idx)
            else:
                seen_hashes[text_hash] = [idx]

        # Also check for near-exact using character similarity
        seen_fingerprints: Dict[str, int] = {}
        for idx, text in enumerate(texts):
            if idx in duplicate_indices:
                continue
            cleaned = " ".join(text.lower().split())
            fingerprint = hashlib.md5(cleaned.encode()).hexdigest()
            if fingerprint in seen_fingerprints:
                orig_idx = seen_fingerprints[fingerprint]
                if (orig_idx, idx, 1.0) not in duplicates:
                    duplicates.append((orig_idx, idx, 1.0))
                    duplicate_indices.add(idx)
            else:
                seen_fingerprints[fingerprint] = idx

        examples = []
        for orig_idx, dup_idx, sim in duplicates[:10]:
            examples.append(f"  [{orig_idx}] ≈ [{dup_idx}] (sim={sim:.3f})")

        return {
            "duplicates": duplicates,
            "duplicate_count": len(duplicates),
            "duplicate_indices": list(duplicate_indices),
            "examples": examples,
        }

    def find_near_duplicates(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
        max_texts: int = 10000,
    ) -> Dict[str, Any]:
        """Find near-duplicate texts using character n-gram similarity.

        Uses minhash-like approach with character shingles for efficiency.

        Args:
            texts: List of text strings.
            threshold: Similarity threshold (0-1).
            max_texts: Maximum texts to compare.

        Returns:
            Dictionary with near-duplicate pairs.
        """
        threshold = threshold or self._near_dup_thresh
        texts = texts[:max_texts]

        # Compute character n-gram sets
        ngram_sets = []
        for text in texts:
            cleaned = text.lower().strip()
            ngrams = set(_char_ngrams(cleaned, 3))
            ngram_sets.append(ngrams)

        # Compare pairs
        pairs = []
        examples = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not ngram_sets[i] and not ngram_sets[j]:
                    sim = 1.0
                elif not ngram_sets[i] or not ngram_sets[j]:
                    sim = 0.0
                else:
                    sim = _jaccard_similarity(ngram_sets[i], ngram_sets[j])

                if sim >= threshold:
                    pairs.append((i, j, sim))
                    if len(examples) < 10:
                        examples.append(
                            f"  [{i}] ≈ [{j}] (sim={sim:.3f})"
                        )

        return {
            "pairs": pairs,
            "pair_count": len(pairs),
            "threshold": threshold,
            "examples": examples,
        }

    def detect_anomalies(
        self,
        texts: List[str],
        method: str = "length",
    ) -> List[str]:
        """Detect anomalous texts based on various heuristics.

        Args:
            texts: List of text strings.
            method: Detection method: "length", "character", "token".

        Returns:
            List of anomaly descriptions.
        """
        anomalies = []
        if not texts:
            return anomalies

        if method == "length":
            lengths = [len(t) for t in texts]
            return self._detect_length_anomalies(lengths, texts)
        elif method == "character":
            return self._detect_character_anomalies(texts)
        elif method == "token":
            token_lengths = [len(_simple_tokenize(t)) for t in texts]
            return self._detect_length_anomalies(token_lengths, texts)
        else:
            # Combine all methods
            anomalies.extend(self._detect_length_anomalies([len(t) for t in texts], texts))
            anomalies.extend(self._detect_character_anomalies(texts))
            return list(set(anomalies))

    def language_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Detect language distribution across texts.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary mapping language code -> proportion.
        """
        lang_counts = Counter()
        sample_size = min(len(texts), 5000)

        for text in texts[:sample_size]:
            lang = _detect_language_simple(text)
            lang_counts[lang] += 1

        total = sum(lang_counts.values())
        if total == 0:
            return {}

        return {lang: count / total for lang, count in lang_counts.most_common()}

    def format_report(self, report: DataQualityReport) -> str:
        """Format a quality report as a readable string.

        Args:
            report: DataQualityReport.

        Returns:
            Multi-line formatted report.
        """
        lines = []
        w = 66
        lines.append(f"{BOX_TL}{'Data Quality Report':^{w}}{BOX_TR}")
        lines.append(
            f"{BOX_V} Documents: {report.total_documents:,}"
            f"{'':>{w - 14 - len(str(report.total_documents))}}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Overall Score: {report.overall_score:.1%}"
            f"{'':>{w - 18}}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * w}{BOX_V}")

        # Summary stats
        lines.append(
            f"{BOX_V} Exact duplicates:    {report.duplicate_count:>8,}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Near duplicates:     {report.near_duplicate_count:>8,}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Anomalies:           {report.anomaly_count:>8,}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Empty documents:     {report.empty_count:>8,}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * w}{BOX_V}")

        # Issues
        if report.issues:
            severity_order = {"error": 0, "warning": 1, "info": 2}
            sorted_issues = sorted(report.issues, key=lambda x: severity_order.get(x.severity, 3))

            for issue in sorted_issues:
                marker = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "?")
                desc = f"[{marker}] {issue.description}"
                if len(desc) > w - 2:
                    desc = desc[: w - 5] + "..."
                lines.append(f"{BOX_V}{desc:<{w}}{BOX_V}")

                for example in issue.examples[:3]:
                    if len(example) > w - 4:
                        example = example[: w - 7] + "..."
                    lines.append(f"{BOX_V} {example:<{w - 2}}{BOX_V}")
        else:
            lines.append(f"{BOX_V}{'No quality issues detected':^{w}}{BOX_V}")

        lines.append(BOX_BL + BOX_H * w + BOX_BR)
        return "\n".join(lines)

    def _detect_length_anomalies(
        self, lengths: List[int], texts: List[str]
    ) -> List[str]:
        """Detect texts with anomalous lengths.

        Args:
            lengths: Length of each text.
            texts: Original texts.

        Returns:
            List of anomaly descriptions.
        """
        anomalies = []
        if not lengths:
            return anomalies

        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)

        # Use IQR method
        q1 = _compute_percentile(sorted_lengths, 25)
        q3 = _compute_percentile(sorted_lengths, 75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 3 * iqr)
        upper_bound = q3 + 3 * iqr

        for idx, length in enumerate(lengths):
            if length > upper_bound:
                preview = texts[idx][:50] if idx < len(texts) else ""
                anomalies.append(
                    f"[{idx}] Very long: {length} chars (threshold: {upper_bound})"
                    f" | {preview}..."
                )
            elif length < lower_bound and length > 0:
                anomalies.append(
                    f"[{idx}] Very short: {length} chars (threshold: {lower_bound})"
                )

        return anomalies

    def _detect_character_anomalies(self, texts: List[str]) -> List[str]:
        """Detect texts with anomalous character patterns.

        Args:
            texts: List of texts.

        Returns:
            List of anomaly descriptions.
        """
        anomalies = []
        for idx, text in enumerate(texts[:10000]):
            if not text:
                continue

            # Check for very high ratio of special characters
            alpha_count = sum(1 for c in text if c.isalpha())
            total_count = len(text)
            if total_count > 50 and alpha_count / total_count < 0.3:
                anomalies.append(
                    f"[{idx}] Low alphabetic ratio: {alpha_count/total_count:.1%}"
                )

            # Check for repetitive patterns
            if len(text) > 20:
                repeated = re.findall(r'(.)\1{10,}', text)
                if repeated:
                    anomalies.append(
                        f"[{idx}] Repetitive characters: {repeated[:3]}"
                    )

            # Check for mostly numbers
            digit_count = sum(1 for c in text if c.isdigit())
            if total_count > 50 and digit_count / total_count > 0.8:
                anomalies.append(
                    f"[{idx}] Mostly numeric: {digit_count/total_count:.1%}"
                )

        return anomalies

    def _check_label_consistency(
        self, texts: List[str], labels: List[str]
    ) -> List[QualityIssue]:
        """Check for label-related quality issues.

        Args:
            texts: Text strings.
            labels: Labels per text.

        Returns:
            List of QualityIssue objects.
        """
        issues = []
        label_counts = Counter(labels)

        # Very rare classes
        total = len(labels)
        for label, count in label_counts.items():
            if count < 5:
                issues.append(QualityIssue(
                    issue_type="rare_class",
                    severity="warning",
                    description=f"Class '{label}' has only {count} samples ({count/total:.2%})",
                    count=count,
                ))

        return issues


# ============================================================================
# DataBalanceChecker
# ============================================================================

class DataBalanceChecker:
    """
    Check class/category balance in a dataset and recommend sampling weights.

    Analyzes label distribution, computes imbalance ratios, and
    generates sampling weights for balanced training.

    Example:
        checker = DataBalanceChecker()
        balance = checker.check_balance(labels)
        print(checker.format_report(balance))
        weights = checker.compute_sampling_weights(labels)
    """

    def __init__(
        self,
        min_samples_per_class: int = 10,
        imbalance_warning_threshold: float = 5.0,
    ):
        """Initialize the balance checker.

        Args:
            min_samples_per_class: Minimum samples for a class to be included.
            imbalance_warning_threshold: Ratio above which imbalance is flagged.
        """
        self._min_samples = min_samples_per_class
        self._imbalance_warn = imbalance_warning_threshold

    def check_balance(
        self,
        labels: List[str],
        weights: Optional[List[float]] = None,
    ) -> List[ClassBalanceInfo]:
        """Check class balance in the dataset.

        Args:
            labels: List of class labels.
            weights: Optional sample weights.

        Returns:
            List of ClassBalanceInfo objects.
        """
        total = len(labels)
        if total == 0:
            return []

        label_counts = Counter(labels)
        num_classes = len(label_counts)
        ideal_proportion = 1.0 / num_classes if num_classes > 0 else 0

        balance_info = []
        max_count = max(label_counts.values()) if label_counts else 1

        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            proportion = count / total
            imbalance_ratio = max_count / count if count > 0 else float("inf")

            # Compute sampling weight (inverse frequency)
            if count > 0:
                sampling_weight = total / (num_classes * count)
            else:
                sampling_weight = 0.0

            info = ClassBalanceInfo(
                label=str(label),
                count=count,
                proportion=proportion,
                ideal_proportion=ideal_proportion,
                imbalance_ratio=imbalance_ratio,
                sampling_weight=sampling_weight,
            )
            balance_info.append(info)

        return balance_info

    def compute_sampling_weights(
        self,
        labels: List[str],
        method: str = "inverse",
        smoothing: float = 0.0,
    ) -> List[float]:
        """Compute per-sample weights for balanced sampling.

        Args:
            labels: List of class labels.
            method: Weighting method: "inverse", "sqrt_inverse", "effective".
            smoothing: Label smoothing parameter.

        Returns:
            List of weights, one per sample.
        """
        balance_info = self.check_balance(labels)
        label_to_weight = {info.label: info.sampling_weight for info in balance_info}

        if method == "inverse":
            return [label_to_weight.get(label, 1.0) for label in labels]

        elif method == "sqrt_inverse":
            return [
                math.sqrt(label_to_weight.get(label, 1.0)) for label in labels
            ]

        elif method == "effective":
            # Effective number of samples (from "Class-Balanced Loss" paper)
            beta = 0.999
            label_counts = Counter(labels)
            num_classes = len(label_counts)
            effective_num = {}
            for label, count in label_counts.items():
                effective_num[label] = (1 - beta ** count) / (1 - beta)

            total_effective = sum(effective_num.values())
            weights = []
            for label in labels:
                en = effective_num.get(label, 1)
                weight = total_effective / (num_classes * en)
                weights.append(weight)
            return weights

        elif method == "log_inverse":
            label_counts = Counter(labels)
            max_count = max(label_counts.values())
            weights = []
            for label in labels:
                count = label_counts.get(label, 1)
                weight = math.log(max_count + 1) / math.log(count + 1)
                weights.append(weight)
            return weights

        else:
            return [1.0] * len(labels)

    def format_report(
        self,
        balance_info: List[ClassBalanceInfo],
        title: str = "Class Balance Report",
    ) -> str:
        """Format a balance report.

        Args:
            balance_info: List of ClassBalanceInfo.
            title: Report title.

        Returns:
            Multi-line report string.
        """
        if not balance_info:
            return "(no data)"

        w = 80
        lines = []
        lines.append(f"{BOX_TL}{title:^{w - 2}}{BOX_TR}")
        lines.append(
            f"{BOX_V}{'Label':<20}{'Count':>10}{'Proportion':>12}{'Imbalance':>12}"
            f"{'Weight':>10}{'Status':>10}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{BOX_H*20}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*12}"
            f"{BOX_CROSS}{BOX_H*12}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*10}{BOX_V}"
        )

        for info in balance_info:
            status = ""
            if info.count < self._min_samples:
                status = "⚠ rare"
            elif info.imbalance_ratio > self._imbalance_warn:
                status = "⚠ imbalanced"
            else:
                status = "✓ ok"

            label = info.label[:18] if len(info.label) > 18 else info.label
            lines.append(
                f"{BOX_V}{label:<20}{info.count:>10,}{info.proportion:>11.2%}"
                f"{info.imbalance_ratio:>11.1f}x{info.sampling_weight:>10.3f}{status:>10}{BOX_V}"
            )

        lines.append(
            f"{BOX_V}{BOX_H*20}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*12}"
            f"{BOX_CROSS}{BOX_H*12}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*10}{BOX_V}"
        )

        # Summary
        total = sum(info.count for info in balance_info)
        num_classes = len(balance_info)
        max_ratio = max(info.imbalance_ratio for info in balance_info)
        min_count = min(info.count for info in balance_info)
        max_count = max(info.count for info in balance_info)

        lines.append(f"{BOX_V} Classes: {num_classes} | Total: {total:,} | "
                     f"Max imbalance: {max_ratio:.1f}x | Range: {min_count}-{max_count}{BOX_V}")
        lines.append(BOX_BL + BOX_H * (w - 2) + BOX_BR)

        # Recommendations
        lines.append("")
        lines.append("Recommendations:")

        rare_classes = [info for info in balance_info if info.count < self._min_samples]
        imbalanced = [info for info in balance_info if info.imbalance_ratio > self._imbalance_warn]

        if rare_classes:
            lines.append(
                f"  ! {len(rare_classes)} classes have fewer than {self._min_samples} samples"
            )
            for info in rare_classes[:5]:
                lines.append(f"      '{info.label}': {info.count} samples")

        if imbalanced:
            lines.append(
                f"  ! {len(imbalanced)} classes have imbalance ratio > {self._imbalance_warn}x"
            )
            lines.append("      Consider using class weights or oversampling")

        if not rare_classes and not imbalanced:
            lines.append("  ✓ Dataset appears well-balanced")

        return "\n".join(lines)

    def balance_chart(
        self,
        balance_info: List[ClassBalanceInfo],
        width: int = 60,
        height: int = 12,
    ) -> str:
        """Render class balance as a bar chart.

        Args:
            balance_info: List of ClassBalanceInfo.
            width: Chart width.
            height: Chart height.

        Returns:
            Multi-line chart string.
        """
        if not balance_info:
            return "(no data)"

        max_count = max(info.count for info in balance_info)
        if max_count == 0:
            return "(all zero counts)"

        chart_width = width - 25
        lines = []
        lines.append("Class Distribution:")

        for info in balance_info:
            label = info.label[:15] if len(info.label) > 15 else info.label
            bar_len = int(info.count / max_count * chart_width)
            bar = PROGRESS_FULL * bar_len + PROGRESS_EMPTY * (chart_width - bar_len)

            status = "⚠" if info.imbalance_ratio > self._imbalance_warn else "✓"
            lines.append(f"  {status} {label:<15} {bar} {info.count:>6,}")

        # Ideal line
        ideal_count = sum(info.count for info in balance_info) / len(balance_info)
        ideal_bar_len = int(ideal_count / max_count * chart_width)
        ideal_pos = " " * (18 + ideal_bar_len)
        lines.append(f"  {'ideal':<15} {' ' * chart_width} {ideal_count:>6,.0f}")

        return "\n".join(lines)

    def entropy_of_labels(self, labels: List[str]) -> float:
        """Compute the entropy of the label distribution.

        Args:
            labels: List of class labels.

        Returns:
            Entropy in nats.
        """
        counts = Counter(labels)
        total = len(labels)
        if total == 0:
            return 0.0
        probs = [count / total for count in counts.values()]
        return -sum(p * math.log(p) for p in probs if p > 0)

    def max_entropy(self, num_classes: int) -> float:
        """Compute maximum entropy for given number of classes.

        Args:
            num_classes: Number of classes.

        Returns:
            Maximum entropy in nats.
        """
        if num_classes <= 1:
            return 0.0
        return math.log(num_classes)

    def normalized_entropy(self, labels: List[str]) -> float:
        """Compute normalized entropy (0=perfect imbalance, 1=perfect balance).

        Args:
            labels: List of class labels.

        Returns:
            Normalized entropy (0-1).
        """
        h = self.entropy_of_labels(labels)
        num_classes = len(set(labels))
        h_max = self.max_entropy(num_classes)
        return h / h_max if h_max > 0 else 0.0
