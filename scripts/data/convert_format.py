#!/usr/bin/env python3
"""
Convert Between Data Formats - Nexus-LLM
==========================================
Converts training data between different formats (ChatML, Alpaca,
ShareGPT, JSONL) for compatibility with various training pipelines.

Usage:
    python convert_format.py --input data.jsonl --from alpaca --to chatml --output converted/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# --- Conversion functions ---

def alpaca_to_chatml(item: Dict) -> Dict:
    """Convert Alpaca format to ChatML."""
    messages = []

    # Construct the user message from instruction + input
    user_content = item.get("instruction", "")
    if item.get("input", "").strip():
        user_content += f"\n\nInput: {item['input']}"

    messages.append({"role": "user", "content": user_content.strip()})
    messages.append({"role": "assistant", "content": item.get("output", "")})

    return {"messages": messages}


def chatml_to_alpaca(item: Dict) -> Dict:
    """Convert ChatML format to Alpaca."""
    messages = item.get("messages", [])

    instruction = ""
    input_text = ""
    output = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            # Prepend system message to instruction
            instruction = f"{content}\n\n" if content else ""
        elif role == "user":
            instruction += content
        elif role == "assistant":
            output = content

    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output.strip(),
    }


def sharegpt_to_chatml(item: Dict) -> Dict:
    """Convert ShareGPT format to ChatML."""
    role_map = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
    }

    messages = []
    for conv in item.get("conversations", []):
        from_role = conv.get("from", "")
        mapped_role = role_map.get(from_role, from_role)
        messages.append({
            "role": mapped_role,
            "content": conv.get("value", ""),
        })

    return {"messages": messages}


def chatml_to_sharegpt(item: Dict) -> Dict:
    """Convert ChatML format to ShareGPT."""
    role_map = {
        "system": "system",
        "user": "human",
        "assistant": "gpt",
    }

    conversations = []
    for msg in item.get("messages", []):
        role = msg.get("role", "")
        from_role = role_map.get(role, role)
        conversations.append({
            "from": from_role,
            "value": msg.get("content", ""),
        })

    return {"conversations": conversations}


def alpaca_to_sharegpt(item: Dict) -> Dict:
    """Convert Alpaca format to ShareGPT."""
    chatml = alpaca_to_chatml(item)
    return chatml_to_sharegpt(chatml)


def sharegpt_to_alpaca(item: Dict) -> Dict:
    """Convert ShareGPT format to Alpaca."""
    chatml = sharegpt_to_chatml(item)
    return chatml_to_alpaca(chatml)


def alpaca_to_preference(item: Dict) -> Dict:
    """
    Convert Alpaca format to preference data format.
    Note: This creates a synthetic rejected response by truncating.
    In practice, you should use actual preference labels.
    """
    instruction = item.get("instruction", "")
    if item.get("input", "").strip():
        instruction += f"\n\nInput: {item['input']}"
    output = item.get("output", "")

    # Create a synthetic rejected response (truncated version)
    # WARNING: This is a placeholder. Use real preference data in production.
    words = output.split()
    mid = max(len(words) // 2, 1)
    rejected = " ".join(words[:mid])

    return {
        "prompt": instruction.strip(),
        "chosen": output,
        "rejected": rejected + "... [truncated]",
    }


# Conversion matrix
CONVERTERS = {
    ("alpaca", "chatml"): alpaca_to_chatml,
    ("chatml", "alpaca"): chatml_to_alpaca,
    ("sharegpt", "chatml"): sharegpt_to_chatml,
    ("chatml", "sharegpt"): chatml_to_sharegpt,
    ("alpaca", "sharegpt"): alpaca_to_sharegpt,
    ("sharegpt", "alpaca"): sharegpt_to_alpaca,
    ("alpaca", "preference"): alpaca_to_preference,
}


def main():
    parser = argparse.ArgumentParser(description="Convert training data formats for Nexus-LLM")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--from", dest="from_format", type=str, required=True,
                       choices=["alpaca", "chatml", "sharegpt"],
                       help="Source format")
    parser.add_argument("--to", dest="to_format", type=str, required=True,
                       choices=["alpaca", "chatml", "sharegpt", "preference"],
                       help="Target format")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--split-ratio", type=float, default=0.0,
                       help="Split into train/val with this ratio (0 = no split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    # Validate conversion
    if args.from_format == args.to_format:
        logger.error("Source and target formats are the same. Nothing to convert.")
        sys.exit(1)

    converter_key = (args.from_format, args.to_format)
    if converter_key not in CONVERTERS:
        logger.error(f"Conversion from {args.from_format} to {args.to_format} is not supported.")
        logger.info(f"Supported conversions: {list(CONVERTERS.keys())}")
        sys.exit(1)

    converter = CONVERTERS[converter_key]

    # Load data
    logger.info(f"Loading data from {args.input}")
    data = load_jsonl(args.input)
    logger.info(f"Loaded {len(data)} examples")

    # Convert
    logger.info(f"Converting from {args.from_format} to {args.to_format}...")
    converted = []
    errors = 0

    for item in data:
        try:
            converted_item = converter(item)
            converted.append(converted_item)
        except Exception as e:
            logger.warning(f"Error converting item: {e}")
            errors += 1

    logger.info(f"Converted {len(converted)} examples ({errors} errors)")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_{args.to_format}.jsonl")

    # Save (with optional splitting)
    if args.split_ratio > 0:
        import random
        random.seed(args.seed)
        random.shuffle(converted)

        split_idx = int(len(converted) * (1 - args.split_ratio))
        train_data = converted[:split_idx]
        val_data = converted[split_idx:]

        train_path = output_path.replace(".jsonl", "_train.jsonl")
        val_path = output_path.replace(".jsonl", "_val.jsonl")

        save_jsonl(train_data, train_path)
        save_jsonl(val_data, val_path)

        logger.info(f"Train: {len(train_data)} examples -> {train_path}")
        logger.info(f"Validation: {len(val_data)} examples -> {val_path}")
    else:
        save_jsonl(converted, output_path)
        logger.info(f"Output: {len(converted)} examples -> {output_path}")

    # Print a sample
    if converted:
        logger.info(f"\nSample converted example:")
        sample = converted[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:500])

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
