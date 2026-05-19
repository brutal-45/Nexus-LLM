"""Dataset handling: load JSONL/CSV/JSON, streaming, train/val split, tokenization."""

import os
import json
import csv
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator, Tuple, Union

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset loading and processing."""
    data_path: str = ""
    data_format: str = "jsonl"
    text_field: str = "text"
    target_field: Optional[str] = None
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    train_split: float = 0.9
    val_split: float = 0.1
    seed: int = 42
    streaming: bool = False
    buffer_size: int = 10000
    num_samples: Optional[int] = None
    prompt_field: Optional[str] = None
    response_field: Optional[str] = None
    system_prompt: Optional[str] = None
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    preprocess_num_workers: int = 0


class TextDataset(Dataset):
    """Dataset for text data loaded from various file formats."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None,
        config: Optional[DataConfig] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config or DataConfig()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return self._tokenize_item(item)

    def _tokenize_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            text = item.get(self.config.text_field, "")
            return {
                "input_ids": torch.tensor([0], dtype=torch.long),
                "text": text,
            }

        text = item.get(self.config.text_field, "")

        if self.config.prompt_field and self.config.response_field:
            prompt = item.get(self.config.prompt_field, "")
            response = item.get(self.config.response_field, "")
            if self.config.system_prompt:
                text = f"{self.config.system_prompt}\n{prompt}\n{response}"
            else:
                text = f"{prompt}\n{response}"

        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.config.target_field and self.config.target_field in item:
            target_text = item[self.config.target_field]
            target_encoded = self.tokenizer(
                target_text,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            result["labels"] = target_encoded["input_ids"].squeeze(0)
        else:
            result["labels"] = result["input_ids"].clone()
            if self.tokenizer and hasattr(self.tokenizer, "pad_token_id"):
                result["labels"][result["labels"] == self.tokenizer.pad_token_id] = -100

        return result


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large-scale data that doesn't fit in memory."""

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        config: Optional[DataConfig] = None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        self._format = self._detect_format(data_path)

    def _detect_format(self, path: str) -> str:
        if path.endswith(".jsonl"):
            return "jsonl"
        elif path.endswith(".csv"):
            return "csv"
        elif path.endswith(".json"):
            return "json"
        return self.config.data_format

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if self._format == "jsonl":
            yield from self._iter_jsonl(worker_info)
        elif self._format == "csv":
            yield from self._iter_csv(worker_info)
        elif self._format == "json":
            yield from self._iter_json(worker_info)
        else:
            raise ValueError(f"Unsupported format for streaming: {self._format}")

    def _iter_jsonl(self, worker_info: Optional[Any] = None) -> Iterator[Dict[str, torch.Tensor]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if worker_info is not None and line_idx % worker_info.num_workers != worker_info.id:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield self._process_item(item)
                except json.JSONDecodeError:
                    continue

    def _iter_csv(self, worker_info: Optional[Any] = None) -> Iterator[Dict[str, torch.Tensor]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                if worker_info is not None and row_idx % worker_info.num_workers != worker_info.id:
                    continue
                yield self._process_item(row)

    def _iter_json(self, worker_info: Optional[Any] = None) -> Iterator[Dict[str, torch.Tensor]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if worker_info is not None and idx % worker_info.num_workers != worker_info.id:
                        continue
                    yield self._process_item(item)

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            return {"text": item.get(self.config.text_field, "")}

        text = item.get(self.config.text_field, "")
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone(),
        }


def load_jsonl(path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line: {e}")
            if num_samples is not None and len(data) >= num_samples:
                break
    return data


def load_csv(path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from a CSV file."""
    data = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
            if num_samples is not None and len(data) >= num_samples:
                break
    return data


def load_json(path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "items", "records", "samples"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        data = [data]
    if num_samples is not None:
        data = data[:num_samples]
    return data


def train_val_split(
    data: List[Dict[str, Any]],
    train_split: float = 0.9,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training and validation sets."""
    rng = random.Random(seed)
    rng.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]

    logger.info(f"Split data: {n_train} train, {n_val} validation")
    return train_data, val_data


def load_dataset(
    data_path: str,
    config: Optional[DataConfig] = None,
    tokenizer: Optional[Any] = None,
) -> Union[TextDataset, StreamingTextDataset, Tuple[TextDataset, TextDataset]]:
    """Load a dataset from the given path.

    Returns a single dataset if streaming, or a (train, val) split otherwise.
    """
    config = config or DataConfig(data_path=data_path)

    if config.streaming:
        return StreamingTextDataset(data_path, tokenizer=tokenizer, config=config)

    fmt = config.data_format
    if fmt == "auto":
        if data_path.endswith(".jsonl"):
            fmt = "jsonl"
        elif data_path.endswith(".csv"):
            fmt = "csv"
        elif data_path.endswith(".json"):
            fmt = "json"
        else:
            fmt = "jsonl"

    if fmt == "jsonl":
        data = load_jsonl(data_path, num_samples=config.num_samples)
    elif fmt == "csv":
        data = load_csv(data_path, num_samples=config.num_samples)
    elif fmt == "json":
        data = load_json(data_path, num_samples=config.num_samples)
    else:
        raise ValueError(f"Unsupported data format: {fmt}")

    logger.info(f"Loaded {len(data)} samples from {data_path}")

    train_data, val_data = train_val_split(
        data, train_split=config.train_split, val_split=config.val_split, seed=config.seed
    )

    train_dataset = TextDataset(train_data, tokenizer=tokenizer, config=config)
    val_dataset = TextDataset(val_data, tokenizer=tokenizer, config=config)

    return train_dataset, val_dataset
