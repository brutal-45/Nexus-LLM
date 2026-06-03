#!/usr/bin/env python3
"""
Prepare Training Dataset - Nexus-LLM
======================================
Prepares raw data for fine-tuning by cleaning, formatting,
and splitting into train/validation sets.

Usage:
    python prepare_dataset.py --input raw_data.jsonl --output prepared_data/
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training dataset for Nexus-LLM fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--format", type=str, default="chatml",
                       choices=["chatml", "alpaca", "sharegpt", "jsonl"],
                       help="Output format (default: chatml)")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1)")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length in tokens (default: 2048)")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum sequence length in tokens (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate examples")
    parser.add_argument("--clean-html", action="store_true", help="Strip HTML tags from text")
    parser.add_argument("--filter-language", type=str, default=None,
                       help="Keep only examples in this language code (e.g., 'en')")
    return parser.parse_args()


def load_data(input_path: str) -> List[Dict]:
    """Load data from a JSONL file or directory of JSONL files."""
    path = Path(input_path)
    data = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    for file_path in files:
        logger.info(f"Loading {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    item["_source_file"] = str(file_path)
                    item["_line_number"] = line_num
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON in {file_path}:{line_num}: {e}")

    logger.info(f"Loaded {len(data)} examples from {len(files)} files")
    return data


def clean_text(text: str, strip_html: bool = False) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove HTML tags if requested
    if strip_html:
        text = re.sub(r"<[^>]+>", "", text)

    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text


def validate_example(example: Dict, min_length: int) -> bool:
    """Validate a single training example."""
    # Check required fields
    if "instruction" not in example and "messages" not in example:
        return False
    if "output" not in example and "messages" not in example:
        return False

    # Check minimum length
    text = example.get("instruction", "") + example.get("output", "")
    if len(text.split()) < min_length:
        return False

    return True


def deduplicate(data: List[Dict]) -> List[Dict]:
    """Remove duplicate examples based on instruction content."""
    seen = set()
    unique = []
    for item in data:
        key = item.get("instruction", "") + item.get("output", "")
        key_hash = hash(key)
        if key_hash not in seen:
            seen.add(key_hash)
            unique.append(item)
    return unique


def format_chatml(example: Dict) -> Dict:
    """Format an example in ChatML format."""
    messages = []

    if "system" in example:
        messages.append({"role": "system", "content": example["system"]})

    instruction = example.get("instruction", "")
    messages.append({"role": "user", "content": instruction})

    output = example.get("output", "")
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def format_alpaca(example: Dict) -> Dict:
    """Format an example in Alpaca format."""
    result = {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "output": example.get("output", ""),
    }
    return result


def format_sharegpt(example: Dict) -> Dict:
    """Format an example in ShareGPT format."""
    conversations = []

    if "system" in example:
        conversations.append({"from": "system", "value": example["system"]})

    conversations.append({"from": "human", "value": example.get("instruction", "")})
    conversations.append({"from": "gpt", "value": example.get("output", "")})

    return {"conversations": conversations}


FORMATTERS = {
    "chatml": format_chatml,
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
    "jsonl": lambda x: x,  # Pass through
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 characters for English)."""
    return len(text) // 4


def main():
    args = parse_args()

    # Load raw data
    data = load_data(args.input)

    # Clean text
    logger.info("Cleaning text...")
    for item in data:
        for key in ["instruction", "output", "system", "input"]:
            if key in item and isinstance(item[key], str):
                item[key] = clean_text(item[key], strip_html=args.clean_html)

    # Validate
    logger.info("Validating examples...")
    original_count = len(data)
    data = [item for item in data if validate_example(item, args.min_length)]
    logger.info(f"Filtered {original_count - len(data)} invalid examples, {len(data)} remaining")

    # Deduplicate
    if args.deduplicate:
        logger.info("Deduplicating...")
        original_count = len(data)
        data = deduplicate(data)
        logger.info(f"Removed {original_count - len(data)} duplicates, {len(data)} remaining")

    # Filter by token length
    logger.info("Filtering by token length...")
    original_count = len(data)
    data = [
        item for item in data
        if args.min_length <= estimate_tokens(str(item)) <= args.max_length
    ]
    logger.info(f"Filtered {original_count - len(data)} by length, {len(data)} remaining")

    # Format
    logger.info(f"Formatting as {args.format}...")
    formatter = FORMATTERS[args.format]
    formatted_data = [formatter(item) for item in data]

    # Split into train/validation
    import random
    random.seed(args.seed)
    random.shuffle(formatted_data)

    split_idx = int(len(formatted_data) * (1 - args.validation_split))
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]

    logger.info(f"Train: {len(train_data)} examples")
    logger.info(f"Validation: {len(val_data)} examples")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    for path, dataset in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(dataset)} examples to {path}")

    # Save metadata
    metadata = {
        "total_examples": len(formatted_data),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "format": args.format,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "deduplicated": args.deduplicate,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset preparation complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
