"""Data collation: padding, truncation, label masking, dynamic batching."""

import logging
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import default_collate

logger = logging.getLogger(__name__)


class DataCollator:
    """Collates batches of data with padding, truncation, and label masking."""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        label_pad_token_id: int = -100,
        truncation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.label_pad_token_id = label_pad_token_id
        self.truncation = truncation

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of features into a batch."""
        if len(features) == 0:
            return {}

        first = features[0]
        batch_keys = set(first.keys())

        has_labels = "labels" in batch_keys
        has_input_ids = "input_ids" in batch_keys

        if has_input_ids and isinstance(first["input_ids"], torch.Tensor):
            return self._collate_tensor_batches(features, has_labels)

        if has_input_ids and isinstance(first["input_ids"], list):
            return self._collate_list_batches(features, has_labels)

        return self._collate_generic(features)

    def _collate_tensor_batches(
        self, features: List[Dict[str, Any]], has_labels: bool
    ) -> Dict[str, torch.Tensor]:
        """Collate batches where values are tensors (variable length)."""
        batch = {}

        input_ids_list = [f["input_ids"] for f in features]

        if self.padding:
            max_len = max(ids.shape[0] for ids in input_ids_list)
            if self.max_length is not None:
                max_len = min(max_len, self.max_length)
            if self.pad_to_multiple_of is not None:
                max_len = ((max_len + self.pad_to_multiple_of - 1) //
                           self.pad_to_multiple_of * self.pad_to_multiple_of)
        else:
            max_len = max(ids.shape[0] for ids in input_ids_list)

        pad_token_id = 0
        if self.tokenizer and hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []

        for ids in input_ids_list:
            ids = ids[:max_len] if self.truncation and ids.shape[0] > max_len else ids
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
            )
            attention = torch.cat([
                torch.ones(ids.shape[0], dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
            padded_attention_mask.append(attention)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(padded_attention_mask)

        if has_labels:
            labels_list = [f["labels"] for f in features]
            padded_labels = []
            for labels in labels_list:
                labels = labels[:max_len] if self.truncation and labels.shape[0] > max_len else labels
                pad_len = max_len - labels.shape[0]
                padded_labels.append(
                    torch.cat([labels, torch.full((pad_len,), self.label_pad_token_id, dtype=labels.dtype)])
                )
            batch["labels"] = torch.stack(padded_labels)

        for key in features[0]:
            if key not in ("input_ids", "attention_mask", "labels"):
                values = [f[key] for f in features]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values

        return batch

    def _collate_list_batches(
        self, features: List[Dict[str, Any]], has_labels: bool
    ) -> Dict[str, torch.Tensor]:
        """Collate batches where values are lists."""
        batch = {}

        for key in features[0]:
            values = [f[key] for f in features]
            if key == "input_ids":
                max_len = max(len(v) for v in values)
                if self.max_length:
                    max_len = min(max_len, self.max_length)

                pad_token_id = 0
                if self.tokenizer and hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                    pad_token_id = self.tokenizer.pad_token_id

                padded = []
                masks = []
                for v in values:
                    v = v[:max_len]
                    pad_len = max_len - len(v)
                    padded.append(v + [pad_token_id] * pad_len)
                    masks.append([1] * len(v) + [0] * pad_len)

                batch["input_ids"] = torch.tensor(padded, dtype=torch.long)
                batch["attention_mask"] = torch.tensor(masks, dtype=torch.long)

            elif key == "labels" and has_labels:
                max_len = batch["input_ids"].shape[1]
                padded = []
                for v in values:
                    v = v[:max_len]
                    pad_len = max_len - len(v)
                    padded.append(v + [self.label_pad_token_id] * pad_len)
                batch["labels"] = torch.tensor(padded, dtype=torch.long)

            elif key == "attention_mask":
                continue
            else:
                if isinstance(values[0], (list, tuple)):
                    try:
                        batch[key] = torch.tensor(values)
                    except (ValueError, TypeError):
                        batch[key] = values
                else:
                    try:
                        batch[key] = torch.tensor(values)
                    except (ValueError, TypeError):
                        batch[key] = values

        if "attention_mask" not in batch and "input_ids" in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        return batch

    def _collate_generic(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generic collation for non-tensor data."""
        batch = {}
        for key in features[0]:
            values = [f[key] for f in features]
            if isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
            else:
                batch[key] = values
        return batch


class DynamicBatchCollator:
    """Data collator that creates dynamically sized batches based on total token count."""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_tokens_per_batch: int = 4096,
        max_length: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self._base_collator = DataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
            label_pad_token_id=label_pad_token_id,
        )

    def group_into_batches(self, features: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group features into batches by total token count."""
        if not features:
            return []

        sorted_features = sorted(
            features,
            key=lambda f: len(f.get("input_ids", [])),
        )

        batches = []
        current_batch = []
        current_max_len = 0

        for feature in sorted_features:
            feat_len = len(feature.get("input_ids", []))
            if self.max_length:
                feat_len = min(feat_len, self.max_length)

            if current_batch:
                new_total = (len(current_batch) + 1) * max(current_max_len, feat_len)
                if new_total > self.max_tokens_per_batch:
                    batches.append(current_batch)
                    current_batch = [feature]
                    current_max_len = feat_len
                else:
                    current_batch.append(feature)
                    current_max_len = max(current_max_len, feat_len)
            else:
                current_batch.append(feature)
                current_max_len = feat_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features - groups by length for efficiency."""
        return self._base_collator(features)
