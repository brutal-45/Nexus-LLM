"""Data preprocessing: cleaning, deduplication, length filtering, tokenization, format conversion."""

import os
import re
import json
import csv
import logging
import hashlib
from typing import Optional, List, Dict, Any, Set, Tuple, Callable

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline for LLM training data."""

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 512,
        remove_duplicates: bool = True,
        normalize_whitespace: bool = True,
        remove_urls: bool = False,
        remove_html: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False,
        language_filter: Optional[str] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.language_filter = language_filter
        self._seen_hashes: Set[str] = set()

    def clean_text(self, text: str) -> str:
        """Apply text cleaning transformations."""
        if not text:
            return ""

        if self.remove_html:
            text = self._remove_html_tags(text)

        if self.remove_urls:
            text = self._remove_urls(text)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.remove_special_chars:
            text = self._remove_special_chars(text)

        if self.lowercase:
            text = text.lower()

        text = text.strip()
        return text

    @staticmethod
    def _remove_html_tags(text: str) -> str:
        """Remove HTML tags from text."""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#\d+;", "", text)
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text."""
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Remove special characters, keeping alphanumeric, spaces, and basic punctuation."""
        text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
        return text

    def filter_by_length(self, text: str, count_unit: str = "characters") -> bool:
        """Filter text by length constraints.

        Args:
            text: Input text.
            count_unit: Either 'characters' or 'words' for counting.

        Returns:
            True if text passes length filter.
        """
        if count_unit == "words":
            length = len(text.split())
        else:
            length = len(text)

        return self.min_length <= length <= self.max_length

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate using hash-based deduplication."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(text_hash)
        return False

    def deduplicate_exact(self, texts: List[str]) -> List[str]:
        """Remove exact duplicates from a list of texts."""
        seen: Set[str] = set()
        unique = []
        for text in texts:
            normalized = text.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(text)
        logger.info(f"Deduplication: {len(texts)} -> {len(unique)} texts")
        return unique

    def deduplicate_fuzzy(
        self,
        texts: List[str],
        similarity_threshold: float = 0.9,
        ngram_size: int = 5,
    ) -> List[str]:
        """Remove fuzzy (near) duplicates using n-gram Jaccard similarity."""
        def get_ngrams(text: str, n: int) -> Set[str]:
            words = text.lower().split()
            return set(" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1)))

        unique_texts = []
        unique_ngrams = []

        for text in texts:
            text_ngrams = get_ngrams(text, ngram_size)
            if not text_ngrams:
                unique_texts.append(text)
                unique_ngrams.append(text_ngrams)
                continue

            is_dup = False
            for existing_ngrams in unique_ngrams:
                if not existing_ngrams:
                    continue
                intersection = len(text_ngrams & existing_ngrams)
                union = len(text_ngrams | existing_ngrams)
                if union > 0 and intersection / union >= similarity_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique_texts.append(text)
                unique_ngrams.append(text_ngrams)

        logger.info(f"Fuzzy deduplication: {len(texts)} -> {len(unique_texts)} texts")
        return unique_texts

    def process_item(self, item: Dict[str, Any], text_field: str = "text") -> Optional[Dict[str, Any]]:
        """Process a single data item through the preprocessing pipeline."""
        text = item.get(text_field, "")
        if not text:
            return None

        cleaned_text = self.clean_text(text)

        if not self.filter_by_length(cleaned_text):
            return None

        if self.remove_duplicates and self.is_duplicate(cleaned_text):
            return None

        result = dict(item)
        result[text_field] = cleaned_text
        result["original_length"] = len(text)
        result["cleaned_length"] = len(cleaned_text)

        return result

    def process_dataset(
        self,
        data: List[Dict[str, Any]],
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """Process an entire dataset through the preprocessing pipeline."""
        processed = []
        skipped = 0

        for item in data:
            result = self.process_item(item, text_field)
            if result is not None:
                processed.append(result)
            else:
                skipped += 1

        logger.info(
            f"Preprocessing complete: {len(processed)} kept, "
            f"{skipped} skipped out of {len(data)} total"
        )
        return processed

    def convert_jsonl_to_csv(
        self,
        input_path: str,
        output_path: str,
        fieldnames: Optional[List[str]] = None,
    ):
        """Convert JSONL file to CSV format."""
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        if not data:
            logger.warning("No data found in input file.")
            return

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                writer.writerow({k: item.get(k, "") for k in fieldnames})

        logger.info(f"Converted {len(data)} records from JSONL to CSV: {output_path}")

    def convert_csv_to_jsonl(
        self,
        input_path: str,
        output_path: str,
    ):
        """Convert CSV file to JSONL format."""
        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            for row in reader:
                fout.write(json.dumps(dict(row)) + "\n")
                count += 1

        logger.info(f"Converted {count} records from CSV to JSONL: {output_path}")

    def convert_json_to_jsonl(
        self,
        input_path: str,
        output_path: str,
    ):
        """Convert JSON file to JSONL format."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("data", "items", "records", "samples"):
                if key in data:
                    data = data[key]
                    break
            if isinstance(data, dict):
                data = [data]

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Converted {len(data)} records from JSON to JSONL: {output_path}")

    def tokenize_texts(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> Dict[str, Any]:
        """Tokenize a list of texts using a tokenizer.

        Args:
            texts: List of text strings.
            tokenizer: A tokenizer with __call__ method.
            max_length: Maximum sequence length.
            truncation: Whether to truncate.
            padding: Padding strategy.

        Returns:
            Dictionary with tokenized outputs.
        """
        encoded = tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )
        return encoded

    def get_statistics(self, data: List[Dict[str, Any]], text_field: str = "text") -> Dict[str, Any]:
        """Compute statistics about a dataset."""
        lengths = []
        word_counts = []
        total_chars = 0

        for item in data:
            text = item.get(text_field, "")
            lengths.append(len(text))
            word_counts.append(len(text.split()))
            total_chars += len(text)

        if not lengths:
            return {"num_samples": 0}

        avg_len = sum(lengths) / len(lengths)
        avg_words = sum(word_counts) / len(word_counts)

        sorted_lengths = sorted(lengths)
        median_len = sorted_lengths[len(sorted_lengths) // 2]

        return {
            "num_samples": len(data),
            "total_characters": total_chars,
            "avg_characters": avg_len,
            "median_characters": median_len,
            "min_characters": min(lengths),
            "max_characters": max(lengths),
            "avg_words": avg_words,
            "min_words": min(word_counts),
            "max_words": max(word_counts),
        }

    def reset(self):
        """Reset the preprocessor state (especially the deduplication cache)."""
        self._seen_hashes = set()
