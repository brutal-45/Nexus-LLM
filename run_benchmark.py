#!/usr/bin/env python3
"""Nexus-LLM Benchmark Runner Script.

Benchmark LLM inference performance across different configurations.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def run_inference_benchmark(
    model_name: str,
    device: str,
    batch_sizes: List[int],
    seq_lengths: List[int],
    warmup: int,
    iterations: int,
) -> List[Dict[str, Any]]:
    """Run inference benchmark across different batch sizes and sequence lengths.

    Args:
        model_name: Model name or path.
        device: Device to use.
        batch_sizes: List of batch sizes to test.
        seq_lengths: List of sequence lengths to test.
        warmup: Number of warmup iterations.
        iterations: Number of benchmark iterations.

    Returns:
        List of benchmark result dictionaries.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results: List[Dict[str, Any]] = []

    console = None
    try:
        from rich.console import Console
        console = Console()
        console.print(f"[bold blue]Loading model:[/bold blue] {model_name}")
    except ImportError:
        print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    total_configs = len(batch_sizes) * len(seq_lengths)
    current = 0

    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            current += 1
            config_name = f"batch={batch_size}, seq_len={seq_length}"

            if console:
                console.print(f"\n[{current}/{total_configs}] Benchmarking {config_name}")

            input_ids = torch.randint(
                0, tokenizer.vocab_size, (batch_size, seq_length), device=device
            )
            attention_mask = torch.ones_like(input_ids)

            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Synchronize GPU if available
            if device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            latencies: List[float] = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)

            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            throughput = batch_size / avg_latency

            result = {
                "model": model_name,
                "device": device,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "avg_latency_ms": avg_latency * 1000,
                "min_latency_ms": min_latency * 1000,
                "max_latency_ms": max_latency * 1000,
                "throughput_samples_per_sec": throughput,
                "warmup_iterations": warmup,
                "benchmark_iterations": iterations,
            }
            results.append(result)

            if console:
                console.print(
                    f"  Avg latency: {avg_latency * 1000:.2f}ms | "
                    f"Throughput: {throughput:.2f} samples/sec"
                )
            else:
                print(
                    f"  Avg latency: {avg_latency * 1000:.2f}ms | "
                    f"Throughput: {throughput:.2f} samples/sec"
                )

    # Free memory
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main() -> None:
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Benchmark - Benchmark LLM inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --model gpt2-medium
  python run_benchmark.py --model gpt2-medium --batch-sizes 1,2,4 --seq-lengths 128,256
  python run_benchmark.py --model mistral-7b --output results.json
        """,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEXUS_LLM_DEFAULT_MODEL", "gpt2-medium"),
        help="Model name or path to benchmark (default: gpt2-medium)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("NEXUS_LLM_DEVICE", "auto"),
        help="Device to use: auto/cpu/cuda/mps (default: auto)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes to test (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--seq-lengths",
        default="128,256,512,1024",
        help="Comma-separated sequence lengths to test (default: 128,256,512,1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for benchmark results (JSON)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    seq_lengths = [int(s.strip()) for s in args.seq_lengths.split(",")]

    results = run_inference_benchmark(
        model_name=args.model,
        device=args.device,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")

    # Print summary table
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Benchmark Results Summary")
        table.add_column("Batch Size", justify="right")
        table.add_column("Seq Length", justify="right")
        table.add_column("Avg Latency (ms)", justify="right")
        table.add_column("Throughput (samples/s)", justify="right")

        for r in results:
            table.add_row(
                str(r["batch_size"]),
                str(r["seq_length"]),
                f"{r['avg_latency_ms']:.2f}",
                f"{r['throughput_samples_per_sec']:.2f}",
            )

        console.print(table)
    except ImportError:
        print("\nBenchmark Results:")
        print("-" * 70)
        print(f"{'Batch':>6} {'SeqLen':>8} {'Latency(ms)':>14} {'Throughput':>14}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['batch_size']:>6} {r['seq_length']:>8} "
                f"{r['avg_latency_ms']:>14.2f} {r['throughput_samples_per_sec']:>14.2f}"
            )


if __name__ == "__main__":
    main()
