"""Nexus-LLM Data Validator Module.

Validates LLM training datasets for:

- **Schema validation**: Check required keys and value types.
- **Completeness**: Detect missing, null, or empty fields.
- **Quality scoring**: Heuristic quality score combining multiple checks.
- **Text length**: Flag outliers by character and word count.
- **Duplication detection**: Exact and near-duplicate row detection.

Results are returned as :class:`ValidationResult` objects with detailed
issue lists and aggregate scores.

Example::

    from nexus_llm.data.validator import DataValidator

    validator = DataValidator()
    result = validator.validate(data)
    print(result.summary())
    if not result.is_valid:
        for issue in result.issues:
            print(f"  [{issue['severity']}] {issue['message']}")
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Container for validation outcomes.

    Attributes:
        is_valid: True if no errors (severity ``"error"``) were found.
        score: Aggregate quality score in [0.0, 1.0].
        issues: List of issue dicts with keys ``severity``, ``check``,
            ``message``, and optional ``details``.
        stats: Summary statistics about the dataset.
    """

    is_valid: bool = True
    score: float = 1.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        severity: str,
        check: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a validation issue.

        Args:
            severity: One of ``"error"``, ``"warning"``, ``"info"``.
            check: Name of the check that produced the issue.
            message: Human-readable description.
            details: Optional extra information.
        """
        issue: Dict[str, Any] = {
            "severity": severity,
            "check": check,
            "message": message,
        }
        if details:
            issue["details"] = details
        self.issues.append(issue)
        if severity == "error":
            self.is_valid = False

    def errors(self) -> List[Dict[str, Any]]:
        """Return only issues with severity ``error``."""
        return [i for i in self.issues if i["severity"] == "error"]

    def warnings(self) -> List[Dict[str, Any]]:
        """Return only issues with severity ``warning``."""
        return [i for i in self.issues if i["severity"] == "warning"]

    def summary(self) -> str:
        """Return a human-readable summary."""
        n_errors = len(self.errors())
        n_warnings = len(self.warnings())
        n_info = len(self.issues) - n_errors - n_warnings
        lines = [
            f"ValidationResult(valid={self.is_valid}, score={self.score:.3f})",
            f"  errors:   {n_errors}",
            f"  warnings: {n_warnings}",
            f"  info:     {n_info}",
        ]
        if self.stats:
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_schema(
    data: List[Dict[str, Any]],
    required_keys: List[str],
    key_types: Optional[Dict[str, type]] = None,
) -> List[Dict[str, Any]]:
    """Validate that every row has the required keys with correct types.

    Args:
        data: Dataset rows.
        required_keys: Keys that must be present in every row.
        key_types: Optional ``{key: expected_type}`` mapping.

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        for key in required_keys:
            if key not in row:
                issues.append({
                    "severity": "error",
                    "check": "schema",
                    "message": f"Row {idx} missing required key: {key!r}",
                    "details": {"row_index": idx, "key": key},
                })
            elif key_types and key in key_types:
                expected = key_types[key]
                if not isinstance(row[key], expected):
                    issues.append({
                        "severity": "error",
                        "check": "schema_type",
                        "message": (
                            f"Row {idx} key {key!r} expected type "
                            f"{expected.__name__}, got {type(row[key]).__name__}"
                        ),
                        "details": {"row_index": idx, "key": key},
                    })
    return issues


def check_completeness(
    data: List[Dict[str, Any]],
    check_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Check for null, empty, or whitespace-only field values.

    Args:
        data: Dataset rows.
        check_keys: Keys to check.  If ``None``, checks all keys.

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []
    keys_to_check = check_keys or (
        list(data[0].keys()) if data else []
    )
    missing_counts: Counter = Counter()

    for idx, row in enumerate(data):
        for key in keys_to_check:
            val = row.get(key)
            if val is None:
                missing_counts[key] += 1
                if missing_counts[key] <= 5:  # Report first 5 per key
                    issues.append({
                        "severity": "warning",
                        "check": "completeness",
                        "message": f"Row {idx} has None for key {key!r}",
                        "details": {"row_index": idx, "key": key},
                    })
            elif isinstance(val, str) and not val.strip():
                missing_counts[key] += 1
                if missing_counts[key] <= 5:
                    issues.append({
                        "severity": "warning",
                        "check": "completeness",
                        "message": f"Row {idx} has empty string for key {key!r}",
                        "details": {"row_index": idx, "key": key},
                    })

    # Summary for keys with many missing values
    for key, count in missing_counts.items():
        if count > 5:
            issues.append({
                "severity": "warning",
                "check": "completeness_summary",
                "message": f"Key {key!r} has {count} missing/empty values",
                "details": {"key": key, "count": count},
            })

    return issues


def check_text_length(
    data: List[Dict[str, Any]],
    text_key: str = "text",
    min_chars: int = 1,
    max_chars: int = 100000,
    min_words: int = 0,
    max_words: int = 100000,
    outlier_threshold: float = 3.0,
) -> List[Dict[str, Any]]:
    """Validate text length constraints and flag statistical outliers.

    Args:
        data: Dataset rows.
        text_key: Key containing the text to measure.
        min_chars: Minimum character length.
        max_chars: Maximum character length.
        min_words: Minimum word count.
        max_words: Maximum word count.
        outlier_threshold: Standard-deviation multiplier for outlier detection.

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []
    lengths: List[int] = []
    word_counts: List[int] = []

    for idx, row in enumerate(data):
        text = row.get(text_key, "")
        char_len = len(text)
        word_count = len(text.split()) if text else 0
        lengths.append(char_len)
        word_counts.append(word_count)

        if char_len < min_chars:
            issues.append({
                "severity": "error",
                "check": "text_length",
                "message": f"Row {idx} text too short: {char_len} chars (min={min_chars})",
                "details": {"row_index": idx, "char_length": char_len},
            })
        elif char_len > max_chars:
            issues.append({
                "severity": "error",
                "check": "text_length",
                "message": f"Row {idx} text too long: {char_len} chars (max={max_chars})",
                "details": {"row_index": idx, "char_length": char_len},
            })

        if word_count < min_words:
            issues.append({
                "severity": "warning",
                "check": "word_count",
                "message": f"Row {idx} too few words: {word_count} (min={min_words})",
                "details": {"row_index": idx, "word_count": word_count},
            })
        elif word_count > max_words:
            issues.append({
                "severity": "warning",
                "check": "word_count",
                "message": f"Row {idx} too many words: {word_count} (max={max_words})",
                "details": {"row_index": idx, "word_count": word_count},
            })

    # Outlier detection (only if enough data)
    if len(lengths) >= 10:
        import statistics
        mean_len = statistics.mean(lengths)
        stdev_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
        if stdev_len > 0:
            for idx, char_len in enumerate(lengths):
                z = abs(char_len - mean_len) / stdev_len
                if z > outlier_threshold:
                    issues.append({
                        "severity": "info",
                        "check": "length_outlier",
                        "message": f"Row {idx} is a length outlier (z={z:.1f}, chars={char_len})",
                        "details": {"row_index": idx, "z_score": round(z, 2), "char_length": char_len},
                    })

    return issues


def check_duplicates(
    data: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    similarity_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    """Detect exact and near-duplicate rows.

    Args:
        data: Dataset rows.
        keys: Keys to use for duplicate comparison.  If ``None``, uses
            all keys.
        similarity_threshold: 1.0 for exact dedup only; lower values
            flag near-duplicates based on Jaccard similarity of word
            sets.  Range (0, 1].

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []
    keys = keys or (list(data[0].keys()) if data else [])

    # Exact duplicate detection via hash
    seen_hashes: Dict[str, int] = {}
    duplicate_count = 0

    for idx, row in enumerate(data):
        # Build a string from the selected keys for hashing
        content = "|".join(str(row.get(k, "")) for k in keys)
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            duplicate_count += 1
            if duplicate_count <= 10:  # Report first 10
                issues.append({
                    "severity": "warning",
                    "check": "exact_duplicate",
                    "message": f"Row {idx} is an exact duplicate of row {seen_hashes[h]}",
                    "details": {"row_index": idx, "duplicate_of": seen_hashes[h]},
                })
        else:
            seen_hashes[h] = idx

    if duplicate_count > 10:
        issues.append({
            "severity": "warning",
            "check": "duplicate_summary",
            "message": f"Found {duplicate_count} exact duplicate rows total",
            "details": {"count": duplicate_count},
        })

    # Near-duplicate detection (Jaccard similarity on word sets)
    if similarity_threshold < 1.0 and len(data) <= 10000:
        word_sets: List[Set[str]] = []
        near_dup_count = 0
        for row in data:
            text = " ".join(str(row.get(k, "")) for k in keys).lower()
            word_sets.append(set(text.split()))

        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                if not word_sets[i] or not word_sets[j]:
                    continue
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    sim = intersection / union
                    if sim >= similarity_threshold and sim < 1.0:
                        near_dup_count += 1
                        if near_dup_count <= 10:
                            issues.append({
                                "severity": "info",
                                "check": "near_duplicate",
                                "message": (
                                    f"Rows {i} and {j} are near-duplicates "
                                    f"(similarity={sim:.3f})"
                                ),
                                "details": {
                                    "row_a": i, "row_b": j,
                                    "similarity": round(sim, 3),
                                },
                            })

        if near_dup_count > 10:
            issues.append({
                "severity": "info",
                "check": "near_duplicate_summary",
                "message": f"Found {near_dup_count} near-duplicate pairs total",
                "details": {"count": near_dup_count},
            })

    return issues


def compute_quality_score(
    data: List[Dict[str, Any]],
    text_key: str = "text",
) -> float:
    """Compute a heuristic quality score for the dataset.

    The score is a weighted combination of:
    - Completeness (non-empty text): 30%
    - Average word diversity (unique words / total words): 30%
    - Length reasonableness (penalize very short or very long texts): 20%
    - Absence of excessive repetition: 20%

    Returns:
        Float in [0.0, 1.0].
    """
    if not data:
        return 0.0

    n = len(data)

    # Completeness
    non_empty = sum(1 for row in data if row.get(text_key, "").strip())
    completeness = non_empty / n

    # Word diversity
    all_words: List[str] = []
    lengths: List[int] = []
    for row in data:
        text = row.get(text_key, "")
        words = text.lower().split()
        all_words.extend(words)
        lengths.append(len(text))

    total_words = len(all_words)
    unique_words = len(set(all_words)) if all_words else 0
    diversity = unique_words / total_words if total_words > 0 else 0.0

    # Length reasonableness (penalize < 10 or > 100000 chars)
    import statistics
    avg_len = statistics.mean(lengths) if lengths else 0
    if avg_len < 10:
        length_score = avg_len / 10.0
    elif avg_len > 100000:
        length_score = max(0.0, 1.0 - (avg_len - 100000) / 1000000)
    else:
        length_score = 1.0

    # Repetition check (fraction of unique texts)
    text_set = set()
    for row in data:
        t = row.get(text_key, "").strip()
        if t:
            text_set.add(hashlib.sha256(t.encode("utf-8")).hexdigest())
    unique_ratio = len(text_set) / non_empty if non_empty > 0 else 0.0

    score = (
        0.30 * completeness
        + 0.30 * diversity
        + 0.20 * length_score
        + 0.20 * unique_ratio
    )
    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# DataValidator class
# ---------------------------------------------------------------------------

class DataValidator:
    """Orchestrates multiple validation checks on a dataset.

    Each check produces issues that are aggregated into a single
    :class:`ValidationResult`.  Checks can be individually enabled or
    disabled.

    Args:
        required_keys: Keys that must be present in every row.
        key_types: Expected types for specific keys.
        text_key: Key containing the text field to validate.
        min_chars: Minimum text character length.
        max_chars: Maximum text character length.
        min_words: Minimum word count.
        max_words: Maximum word count.
        duplicate_keys: Keys used for duplicate detection.
        similarity_threshold: Threshold for near-duplicate detection.
        enable_schema: Whether to run schema checks.
        enable_completeness: Whether to run completeness checks.
        enable_text_length: Whether to run text-length checks.
        enable_duplicates: Whether to run duplicate checks.
        enable_quality_score: Whether to compute a quality score.

    Example::

        validator = DataValidator(required_keys=["text", "label"])
        result = validator.validate(my_data)
        if not result.is_valid:
            print(result.summary())
    """

    def __init__(
        self,
        required_keys: Optional[List[str]] = None,
        key_types: Optional[Dict[str, type]] = None,
        text_key: str = "text",
        min_chars: int = 1,
        max_chars: int = 100000,
        min_words: int = 0,
        max_words: int = 100000,
        duplicate_keys: Optional[List[str]] = None,
        similarity_threshold: float = 1.0,
        enable_schema: bool = True,
        enable_completeness: bool = True,
        enable_text_length: bool = True,
        enable_duplicates: bool = True,
        enable_quality_score: bool = True,
    ) -> None:
        self._required_keys = required_keys or []
        self._key_types = key_types
        self._text_key = text_key
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._min_words = min_words
        self._max_words = max_words
        self._duplicate_keys = duplicate_keys
        self._similarity_threshold = similarity_threshold
        self._enable_schema = enable_schema
        self._enable_completeness = enable_completeness
        self._enable_text_length = enable_text_length
        self._enable_duplicates = enable_duplicates
        self._enable_quality_score = enable_quality_score

    # -- Public API ---------------------------------------------------------

    def validate(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Run all enabled validation checks on *data*.

        Args:
            data: Dataset as a list of row dicts.

        Returns:
            Aggregated :class:`ValidationResult`.
        """
        result = ValidationResult(is_valid=True, score=1.0)

        if not data:
            result.add_issue("error", "empty_dataset", "Dataset is empty")
            result.score = 0.0
            return result

        # Stats
        result.stats["num_rows"] = len(data)
        result.stats["columns"] = list(data[0].keys()) if data else []

        # Schema
        if self._enable_schema and self._required_keys:
            for issue in check_schema(data, self._required_keys, self._key_types):
                result.add_issue(**issue)

        # Completeness
        if self._enable_completeness:
            check_keys = self._required_keys or None
            for issue in check_completeness(data, check_keys):
                result.add_issue(**issue)

        # Text length
        if self._enable_text_length:
            for issue in check_text_length(
                data,
                text_key=self._text_key,
                min_chars=self._min_chars,
                max_chars=self._max_chars,
                min_words=self._min_words,
                max_words=self._max_words,
            ):
                result.add_issue(**issue)

        # Duplicates
        if self._enable_duplicates:
            for issue in check_duplicates(
                data,
                keys=self._duplicate_keys,
                similarity_threshold=self._similarity_threshold,
            ):
                result.add_issue(**issue)

        # Quality score
        if self._enable_quality_score:
            result.score = compute_quality_score(data, text_key=self._text_key)

        return result

    def validate_schema(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Run only schema validation."""
        result = ValidationResult()
        if self._required_keys:
            for issue in check_schema(data, self._required_keys, self._key_types):
                result.add_issue(**issue)
        return result

    def validate_completeness(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Run only completeness checks."""
        result = ValidationResult()
        for issue in check_completeness(data, self._required_keys or None):
            result.add_issue(**issue)
        return result

    def validate_text_length(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Run only text-length checks."""
        result = ValidationResult()
        for issue in check_text_length(
            data,
            text_key=self._text_key,
            min_chars=self._min_chars,
            max_chars=self._max_chars,
            min_words=self._min_words,
            max_words=self._max_words,
        ):
            result.add_issue(**issue)
        return result

    def validate_duplicates(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Run only duplicate checks."""
        result = ValidationResult()
        for issue in check_duplicates(
            data, keys=self._duplicate_keys,
            similarity_threshold=self._similarity_threshold,
        ):
            result.add_issue(**issue)
        return result

    def quality_score(self, data: List[Dict[str, Any]]) -> float:
        """Return the heuristic quality score for *data*."""
        return compute_quality_score(data, text_key=self._text_key)
