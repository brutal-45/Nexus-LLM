"""Test data splitting for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


def split_data(data: List[Any], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
               shuffle: bool = True, seed: int = 42) -> Tuple[List, List, List]:
    import random
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


def k_fold_split(data: List[Any], k: int = 5, shuffle: bool = True, seed: int = 42) -> List[Tuple[List, List]]:
    import random
    items = list(data)
    if shuffle:
        random.seed(seed)
        random.shuffle(items)
    fold_size = len(items) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(items)
        val = items[start:end]
        train = items[:start] + items[end:]
        folds.append((train, val))
    return folds


def stratified_split(data: List[Dict], label_key: str = "label", train_ratio: float = 0.8,
                     seed: int = 42) -> Tuple[List, List]:
    import random
    random.seed(seed)
    by_label: Dict[Any, List] = {}
    for item in data:
        label = item.get(label_key)
        by_label.setdefault(label, []).append(item)

    train, val = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train.extend(items[:split_idx])
        val.extend(items[split_idx:])
    return train, val


class DataSplitter:
    def __init__(self, config: SplitConfig = None):
        self._config = config or SplitConfig()

    def split(self, data: List[Any]) -> Tuple[List, List, List]:
        return split_data(
            data,
            train_ratio=self._config.train_ratio,
            val_ratio=self._config.val_ratio,
            test_ratio=self._config.test_ratio,
            shuffle=self._config.shuffle,
            seed=self._config.seed,
        )

    def k_fold(self, data: List[Any], k: int = 5) -> List[Tuple[List, List]]:
        return k_fold_split(data, k, shuffle=self._config.shuffle, seed=self._config.seed)

    def stratified_split(self, data: List[Dict], label_key: str = "label") -> Tuple[List, List]:
        return stratified_split(data, label_key, self._config.train_ratio, self._config.seed)


class TestSplitConfig:
    def test_defaults(self):
        config = SplitConfig()
        assert config.train_ratio == 0.8
        assert config.shuffle is True

    def test_invalid_ratios(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestSplitData:
    def test_split_sizes(self):
        data = list(range(100))
        train, val, test = split_data(data)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_ratios(self):
        data = list(range(100))
        train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        assert len(train) == 70

    def test_no_shuffle(self):
        data = list(range(10))
        train, val, test = split_data(data, shuffle=False)
        assert train == list(range(8))

    def test_invalid_ratios(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_data([], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_all_data_preserved(self):
        data = list(range(50))
        train, val, test = split_data(data)
        assert len(train) + len(val) + len(test) == 50


class TestKFoldSplit:
    def test_k_folds(self):
        data = list(range(100))
        folds = k_fold_split(data, k=5)
        assert len(folds) == 5

    def test_fold_sizes(self):
        data = list(range(100))
        folds = k_fold_split(data, k=5)
        for train, val in folds:
            assert len(val) == 20
            assert len(train) == 80

    def test_all_data_in_each_fold(self):
        data = list(range(50))
        folds = k_fold_split(data, k=5)
        for train, val in folds:
            all_items = set(train) | set(val)
            assert len(all_items) == 50


class TestStratifiedSplit:
    def test_balanced_split(self):
        data = [{"label": 0, "text": f"neg_{i}"} for i in range(50)] + \
               [{"label": 1, "text": f"pos_{i}"} for i in range(50)]
        train, val = stratified_split(data, label_key="label")
        train_labels = [d["label"] for d in train]
        val_labels = [d["label"] for d in val]
        assert 0 in train_labels and 1 in train_labels
        assert 0 in val_labels and 1 in val_labels


class TestDataSplitter:
    def test_split(self):
        splitter = DataSplitter()
        data = list(range(100))
        train, val, test = splitter.split(data)
        assert len(train) + len(val) + len(test) == 100

    def test_k_fold(self):
        splitter = DataSplitter()
        data = list(range(50))
        folds = splitter.k_fold(data, k=5)
        assert len(folds) == 5

    def test_stratified(self):
        splitter = DataSplitter()
        data = [{"label": 0}] * 40 + [{"label": 1}] * 40
        train, val = splitter.stratified_split(data)
        assert len(train) + len(val) == 80
