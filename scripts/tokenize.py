"""
Nexus Tokenizer Training Script (CLI)
==========================================
Usage:
    python -m nexus.scripts.tokenize --train --input data/corpus.jsonl --output tokenizer.json --vocab_size 65536
    python -m nexus.scripts.tokenize --encode --tokenizer tokenizer.json --text "Hello, world!"
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Tokenizer")
    parser.add_argument("--train", action="store_true", help="Train a new tokenizer")
    parser.add_argument("--encode", action="store_true", help="Encode text")
    parser.add_argument("--decode", action="show_true", help="Decode token IDs")
    parser.add_argument("--input", type=str, help="Input file (for training) or text (for encoding)")
    parser.add_argument("--output", type=str, help="Output tokenizer path")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer file (for encode/decode)")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2, help="Min frequency for BPE merges")
    return parser.parse_args()


def main():
    args = parse_args()
    
    from nexus.data.tokenizer import BPETokenizer, TokenizerConfig
    
    if args.train:
        # Train tokenizer
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
        
        # Load training data
        texts = []
        with open(args.input, "r", encoding="utf-8") as f:
            import json
            for line in f:
                try:
                    data = json.loads(line.strip())
                    texts.append(data.get("text", line.strip()))
                except json.JSONDecodeError:
                    texts.append(line.strip())
        
        print(f"Loaded {len(texts)} documents for tokenizer training")
        
        tokenizer = BPETokenizer(
            TokenizerConfig(vocab_size=args.vocab_size)
        )
        tokenizer.train(texts, min_frequency=args.min_frequency)
        
        output_path = args.output or "tokenizer.json"
        tokenizer.save(output_path)
        
        # Test
        test_text = "Hello, world! This is a test of the Nexus tokenizer."
        encoded = tokenizer.encode(test_text, add_bos=False)
        decoded = tokenizer.decode(encoded)
        print(f"\nTest: '{test_text}'")
        print(f"Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''} ({len(encoded)} tokens)")
        print(f"Decoded: '{decoded}'")
        print(f"Roundtrip OK: {test_text == decoded}")
    
    elif args.encode:
        tokenizer = BPETokenizer.load(args.tokenizer)
        encoded = tokenizer.encode(args.input, add_bos=True)
        print(f"Tokens: {encoded}")
        print(f"Token count: {len(encoded)}")
        tokens = tokenizer.tokenize(args.input)
        for i, tok in enumerate(tokens):
            print(f"  {i:3d}: {encoded[i]:6d}  '{tok}'")
    
    else:
        print("Use --train or --encode")
        sys.exit(1)


if __name__ == "__main__":
    main()
