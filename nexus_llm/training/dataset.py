"""Dataset loading and preparation for Nexus-LLM training.

Supports loading from JSONL, JSON, CSV, and HuggingFace datasets with
automatic format detection for Alpaca, Chat, and Instruction styles.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data format enumeration
# ---------------------------------------------------------------------------

class DataFormat(str, Enum):
    """Supported training data formats."""
    ALPACA = "alpaca"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

_ALPACA_KEYS = {"instruction", "input", "output"}
_CHAT_KEYS = {"messages", "role", "content"}
_INSTRUCTION_KEYS = {"prompt", "response", "question", "answer"}


def _detect_format(sample: Dict[str, Any]) -> DataFormat:
    """Auto-detect the data format from a single sample."""
    keys = set(sample.keys())

    if _ALPACA_KEYS.issubset(keys) or keys.issuperset({"instruction", "output"}):
        return DataFormat.ALPACA

    if "messages" in keys or ("role" in keys and "content" in keys):
        return DataFormat.CHAT

    if keys & _INSTRUCTION_KEYS:
        return DataFormat.INSTRUCTION

    # Fallback: try to match by heuristic on the first nested entry
    for v in sample.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return _detect_format(v[0])

    raise TrainingError(
        f"Cannot auto-detect format from keys: {sorted(keys)}. "
        "Specify format explicitly."
    )


# ---------------------------------------------------------------------------
# Format converters – normalise every format to prompt / completion pairs
# ---------------------------------------------------------------------------

def _convert_alpaca(sample: Dict[str, Any]) -> Dict[str, str]:
    """Convert an Alpaca-format sample to prompt/completion."""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")

    if inp:
        prompt = (
            f"Below is an instruction that describes a task, paired with an input "
            f"that provides further context.\n\n### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n### Response:\n"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )
    return {"prompt": prompt, "completion": output}


def _convert_chat(sample: Dict[str, Any]) -> Dict[str, str]:
    """Convert a Chat-format sample to prompt/completion."""
    messages = sample.get("messages", [])
    if not messages:
        raise TrainingError("Chat-format sample has no 'messages' key.")

    prompt_parts: List[str] = []
    completion = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.insert(0, f"<|system|>\n{content}</s>\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}</s>\n")
        elif role == "assistant":
            # Use the last assistant message as completion
            completion = content

    prompt = "".join(prompt_parts) + "<|assistant|)\n"
    return {"prompt": prompt, "completion": completion}


def _convert_instruction(sample: Dict[str, Any]) -> Dict[str, str]:
    """Convert an Instruction-format sample to prompt/completion."""
    prompt = sample.get("prompt") or sample.get("question", "")
    completion = sample.get("response") or sample.get("answer", "")
    return {"prompt": prompt, "completion": completion}


_CONVERTERS = {
    DataFormat.ALPACA: _convert_alpaca,
    DataFormat.CHAT: _convert_chat,
    DataFormat.INSTRUCTION: _convert_instruction,
}


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load samples from a JSONL file."""
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TrainingError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
    return samples


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Load samples from a JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Could be {"data": [...]} or a single sample
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    raise TrainingError(f"Unexpected JSON structure in {path}")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load samples from a CSV file."""
    import csv

    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(dict(row))
    return samples


def _load_hf_dataset(
    dataset_id: str,
    split: str = "train",
    subset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load a dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise TrainingError(
            "The 'datasets' package is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        ) from exc

    kwargs: Dict[str, Any] = {"path": dataset_id, "split": split}
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(**kwargs)
    return [dict(row) for row in ds]


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Load, detect format, normalise, tokenise, and split datasets.

    Usage::

        loader = DatasetLoader(tokenizer=tokenizer)
        train_ds, val_ds = loader.load(
            source="data/train.jsonl",
            format="auto",
            val_split=0.1,
            max_seq_length=512,
        )
    """

    def __init__(self, tokenizer: Any = None) -> None:
        self._tokenizer = tokenizer
        self._format: Optional[DataFormat] = None
        self._raw_samples: List[Dict[str, str]] = []

    @property
    def format(self) -> Optional[DataFormat]:
        """The detected or specified data format."""
        return self._format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        source: str,
        format: str = "auto",
        val_split: float = 0.1,
        max_seq_length: int = 512,
        hf_split: str = "train",
        hf_subset: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load a dataset and return (train, val) splits.

        Args:
            source: Path to a local file (jsonl/json/csv) or HuggingFace
                    dataset ID (e.g. "tatsu-lab/alpaca").
            format: Data format – "alpaca", "chat", "instruction", or "auto".
            val_split: Fraction of data to reserve for validation (0–1).
            max_seq_length: Maximum tokenised sequence length.
            hf_split: Split to use when loading from HuggingFace.
            hf_subset: Optional subset/config for HuggingFace datasets.

        Returns:
            A tuple of (train_dataset, val_dataset) where each element is a
            list of dicts with tokenised ``input_ids`` and ``labels``.
        """
        logger.info("Loading dataset from %s (format=%s)", source, format)

        raw = self._load_raw(source, hf_split=hf_split, hf_subset=hf_subset)
        if not raw:
            raise TrainingError(f"No samples loaded from {source}")

        logger.info("Loaded %d raw samples", len(raw))

        # Detect / validate format
        desired = DataFormat(format)
        if desired == DataFormat.AUTO:
            detected = _detect_format(raw[0])
            logger.info("Auto-detected format: %s", detected.value)
            self._format = detected
        else:
            self._format = desired

        # Convert to prompt/completion pairs
        converter = _CONVERTERS[self._format]
        normalised = [converter(s) for s in raw]
        self._raw_samples = normalised

        # Tokenise
        tokenised = self._tokenise(normalised, max_seq_length)

        # Split
        train_ds, val_ds = self._split(tokenised, val_split)

        logger.info(
            "Dataset ready — train=%d, val=%d (max_seq_length=%d)",
            len(train_ds), len(val_ds), max_seq_length,
        )
        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(
        self,
        source: str,
        hf_split: str = "train",
        hf_subset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load raw samples from a file path or HuggingFace ID."""
        path = Path(source)

        # Try as local file first
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".jsonl":
                return _load_jsonl(path)
            if suffix == ".json":
                return _load_json(path)
            if suffix == ".csv":
                return _load_csv(path)
            raise TrainingError(f"Unsupported file extension: {suffix}")

        # Assume HuggingFace dataset ID
        return _load_hf_dataset(source, split=hf_split, subset=hf_subset)

    def _tokenise(
        self,
        samples: List[Dict[str, str]],
        max_seq_length: int,
    ) -> List[Dict[str, Any]]:
        """Tokenise prompt+completion pairs into input_ids and labels."""
        if self._tokenizer is None:
            logger.warning("No tokenizer provided; skipping tokenisation.")
            return [{"prompt": s["prompt"], "completion": s["completion"]} for s in samples]

        result: List[Dict[str, Any]] = []
        for sample in samples:
            text = sample["prompt"] + sample["completion"]
            enc = self._tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            input_ids = enc["input_ids"]

            # Build labels: mask the prompt portion with -100
            prompt_enc = self._tokenizer(
                sample["prompt"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                return_tensors=None,
            )
            prompt_len = len(prompt_enc["input_ids"])

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            # Pad labels to match input_ids length
            labels = labels[:len(input_ids)]
            labels += [-100] * (len(input_ids) - len(labels))

            result.append({
                "input_ids": input_ids,
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            })

        return result

    @staticmethod
    def _split(
        data: List[Dict[str, Any]],
        val_split: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train and validation sets."""
        if val_split <= 0 or val_split >= 1:
            return data, []

        n_val = max(1, int(len(data) * val_split))
        val_data = data[:n_val]
        train_data = data[n_val:]
        return train_data, val_data
