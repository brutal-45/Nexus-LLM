"""
Nexus Inference Server (CLI)
=================================
Usage:
    python -m nexus.scripts.serve --checkpoint checkpoints/nexus-100b --port 8000
    python -m nexus.scripts.serve --checkpoint checkpoints/nexus-100b --host 0.0.0.0 --port 8080

Launches an OpenAI-compatible inference server.
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Inference Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model_name", type=str, default="nexus-100b", help="Model name for API")
    parser.add_argument("--max_concurrent", type=int, default=512, help="Max concurrent requests")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "int8", "int4"],
                       help="Quantization mode")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    
    from nexus.model.transformer import NexusTransformer
    from nexus.data.tokenizer import BPETokenizer
    from nexus.inference.server import InferenceServer
    
    print(f"\nLoading model from {args.checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = NexusTransformer.from_pretrained(args.checkpoint, device=device)
    
    tokenizer_path = os.path.join(args.checkpoint, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer()
    
    # Optional quantization
    if args.quantization:
        from nexus.inference.quantize import GPTQQuantizer, QuantConfig
        print(f"\nQuantizing model to {args.quantization}...")
        bits = 8 if args.quantization == "int8" else 4
        quantizer = GPTQQuantizer(model, tokenizer, QuantConfig(bits=bits))
        model = quantizer.quantize_model()
    
    # Launch server
    server = InferenceServer(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        max_concurrent_requests=args.max_concurrent,
    )
    
    server.run()


if __name__ == "__main__":
    main()
