"""Nexus-LLM Data Splitter Module.

Provides data splitting strategies for creating train/validation/test
splits and cross-validation folds.  Supported strategies include:

- **Random split**: Simple random shuffle and proportional split.
- **Stratified split**: Preserve class-label proportions across splits.
- **K-fold cross-validation**: Divide data into *k* equally sized folds.
- **Stratified k-fold**: K-fold that preserves label proportions per fold.
- **Time-based split**: Split temporal data by timestamp boundaries.

Both functional helpers and a stateful :class:`DataSplitter` class are
provided.  The class delegates to the functional helpers, making it easy
to swap or extend strategies.

Example::

    from nexus_llm.data.splitter import DataSplitter, SplitConfig

    cfg = SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    splitter = DataSplitter(cfg)
    train, val, test = splitter.split(data)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Configuration for data splitting.

    Attributes:
        train_ratio: Proportion of data for training (default 0.8).
        val_ratio: Proportion of data for validation (default 0.1).
        test_ratio: Proportion of data for testing (default 0.1).
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.
    """

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Functional splitting helpers
# ---------------------------------------------------------------------------

def split_data(
    data: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split *data* into train / validation / test sets.

    Args:
        data: Input sequence to split.
        train_ratio: Fraction for the training set.
        val_ratio: Fraction for the validation set.
        test_ratio: Fraction for the test set.
        shuffle: Shuffle before splitting.
        seed: Random seed.

    Returns:
        ``(train, val, test)`` tuple of lists.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    items = list(data)
    if shuffle:
        random.seed(seed)
        random.shuffle(items)

    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]
    return train, val, test


def k_fold_split(
    data: List[Any],
    k: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Tuple[List[Any], List[Any]]]:
    """Generate *k* train/validation fold pairs.

    Args:
        data: Input sequence.
        k: Number of folds.
        shuffle: Shuffle before folding.
        seed: Random seed.

    Returns:
        List of ``(train_fold, val_fold)`` tuples, one per fold.
    """
    items = list(data)
    if shuffle:
        random.seed(seed)
        random.shuffle(items)
    fold_size = len(items) // k
    folds: List[Tuple[List[Any], List[Any]]] = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(items)
        val = items[start:end]
        train = items[:start] + items[end:]
        folds.append((train, val))
    return folds


def stratified_split(
    data: List[Dict[str, Any]],
    label_key: str = "label",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split *data* while preserving label proportions.

    Args:
        data: Input list of row dicts.
        label_key: Key whose value is used for stratification.
        train_ratio: Fraction for training.
        seed: Random seed.

    Returns:
        ``(train, val)`` tuple.
    """
    random.seed(seed)
    by_label: Dict[Any, List[Dict[str, Any]]] = {}
    for item in data:
        label = item.get(label_key)
        by_label.setdefault(label, []).append(item)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for label, items in by_label.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train.extend(items[:split_idx])
        val.extend(items[split_idx:])
    return train, val


def stratified_k_fold(
    data: List[Dict[str, Any]],
    k: int = 5,
    label_key: str = "label",
    seed: int = 42,
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Stratified k-fold cross-validation.

    Preserves label proportions in each fold.  Each label group is
    divided into *k* roughly equal chunks distributed across folds.

    Args:
        data: Input list of row dicts.
        k: Number of folds.
        label_key: Key for stratification.
        seed: Random seed.

    Returns:
        List of ``(train_fold, val_fold)`` tuples.
    """
    random.seed(seed)
    by_label: Dict[Any, List[Dict[str, Any]]] = {}
    for item in data:
        label = item.get(label_key)
        by_label.setdefault(label, []).append(item)

    # Distribute each label's items across k folds
    fold_data: List[List[Dict[str, Any]]] = [[] for _ in range(k)]
    for label, items in by_label.items():
        random.shuffle(items)
        for idx, item in enumerate(items):
            fold_data[idx % k].append(item)

    # Build train/val pairs
    folds: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for i in range(k):
        val = fold_data[i]
        train = [item for j, fold in enumerate(fold_data) if j != i for item in fold]
        folds.append((train, val))
    return folds


def time_based_split(
    data: List[Dict[str, Any]],
    time_key: str = "timestamp",
    train_until: Optional[Union[str, float]] = None,
    val_until: Optional[Union[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data chronologically based on a timestamp field.

    Rows with ``time_key <= train_until`` go to train, those with
    ``train_until < time_key <= val_until`` go to validation, and the
    rest go to test.

    Args:
        data: Input list of row dicts with a timestamp-like field.
        time_key: Name of the field containing the timestamp.
        train_until: Upper bound (inclusive) for the training set.
        val_until: Upper bound (inclusive) for the validation set.

    Returns:
        ``(train, val, test)`` tuple.

    Raises:
        ValueError: If *train_until* or *val_until* is not provided when
            the other is, or if bounds are inconsistent.
    """
    if val_until is not None and train_until is None:
        raise ValueError("train_until must be provided when val_until is set")
    if train_until is not None and val_until is not None and val_until < train_until:
        raise ValueError("val_until must be >= train_until")

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for row in data:
        t = row.get(time_key)
        if t is None:
            test.append(row)
            continue
        if train_until is not None and t <= train_until:
            train.append(row)
        elif val_until is not None and t <= val_until:
            val.append(row)
        else:
            test.append(row)

    logger.info(
        "Time-based split: train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# DataSplitter class
# ---------------------------------------------------------------------------

class DataSplitter:
    """Stateful data splitter backed by a :class:`SplitConfig`.

    Provides convenient methods that delegate to the functional helpers
    while carrying default configuration (ratios, seed, shuffle).

    Args:
        config: Split configuration.  Defaults to 80/10/10 with shuffling.

    Example::

        splitter = DataSplitter()
        train, val, test = splitter.split(my_data)
        folds = splitter.k_fold(my_data, k=10)
    """

    def __init__(self, config: Optional[SplitConfig] = None) -> None:
        self._config = config or SplitConfig()

    @property
    def config(self) -> SplitConfig:
        """Return the active split configuration."""
        return self._config

    # -- Delegating methods -------------------------------------------------

    def split(
        self, data: List[Any]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Split data into train / val / test using the configured ratios."""
        return split_data(
            data,
            train_ratio=self._config.train_ratio,
            val_ratio=self._config.val_ratio,
            test_ratio=self._config.test_ratio,
            shuffle=self._config.shuffle,
            seed=self._config.seed,
        )

    def k_fold(
        self, data: List[Any], k: int = 5
    ) -> List[Tuple[List[Any], List[Any]]]:
        """Generate *k* cross-validation folds."""
        return k_fold_split(data, k, shuffle=self._config.shuffle, seed=self._config.seed)

    def stratified_split(
        self,
        data: List[Dict[str, Any]],
        label_key: str = "label",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Stratified train/validation split preserving label proportions."""
        return stratified_split(
            data, label_key, self._config.train_ratio, self._config.seed
        )

    def stratified_k_fold(
        self,
        data: List[Dict[str, Any]],
        k: int = 5,
        label_key: str = "label",
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Stratified k-fold cross-validation."""
        return stratified_k_fold(data, k, label_key, self._config.seed)

    def time_based(
        self,
        data: List[Dict[str, Any]],
        time_key: str = "timestamp",
        train_until: Optional[Union[str, float]] = None,
        val_until: Optional[Union[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Chronological train/val/test split."""
        return time_based_split(data, time_key, train_until, val_until)
