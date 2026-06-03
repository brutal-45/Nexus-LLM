#!/usr/bin/env python3
"""
Benchmark Example - Nexus-LLM
================================
Demonstrates how to run performance benchmarks on different models
and configurations.
"""

from nexus_llm import InferenceEngine
from nexus_llm.benchmark import BenchmarkRunner, BenchmarkConfig, BenchmarkSuite


def main():
    # --- Configure the benchmark ---
    config = BenchmarkConfig(
        warmup_iterations=3,          # Warmup runs (not counted)
        benchmark_iterations=10,      # Measured runs
        batch_sizes=[1, 4, 8, 16],   # Batch sizes to test
        sequence_lengths=[128, 256, 512, 1024, 2048],  # Input lengths to test
        max_new_tokens=128,           # Tokens to generate
        metrics=[
            "tokens_per_second",      # Generation throughput
            "time_to_first_token",    # Latency to first token
            "end_to_end_latency",     # Total request latency
            "memory_peak_mb",         # Peak GPU memory usage
            "throughput_requests_per_sec",  # Requests per second
        ],
        output_dir="./benchmark_results",
        device="auto",
    )

    # --- Create the benchmark runner ---
    runner = BenchmarkRunner(config=config)

    # --- Benchmark a single model ---
    print("=" * 60)
    print("Single Model Benchmark")
    print("=" * 60)

    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")
    results = runner.run(engine, model_label="nexus-7b-chat")

    print(f"\nResults for nexus-7b-chat:")
    print(f"  Avg tokens/sec: {results.get_metric('tokens_per_second', aggregate='mean'):.1f}")
    print(f"  Avg TTFT: {results.get_metric('time_to_first_token', aggregate='mean'):.3f}s")
    print(f"  Avg E2E latency: {results.get_metric('end_to_end_latency', aggregate='mean'):.3f}s")
    print(f"  Peak memory: {results.get_metric('memory_peak_mb', aggregate='max'):.0f} MB")

    # --- Compare multiple models ---
    print("\n" + "=" * 60)
    print("Multi-Model Comparison")
    print("=" * 60)

    models = [
        ("nexus-3b-fast", "nexus-3b-chat"),
        ("nexus-7b-chat", "nexus-7b-chat"),
        ("nexus-13b-chat", "nexus-13b-chat"),
    ]

    suite = BenchmarkSuite(config=config)
    for label, model_name in models:
        engine = InferenceEngine(model_name=model_name, device="auto")
        suite.add_benchmark(label, engine)

    comparison = suite.run_all()

    # Print comparison table
    print(f"\n{'Model':<20} {'Tokens/s':<12} {'TTFT (s)':<12} {'Latency (s)':<14} {'Memory (MB)':<14}")
    print("-" * 72)
    for label, results in comparison.items():
        tps = results.get_metric('tokens_per_second', aggregate='mean')
        ttft = results.get_metric('time_to_first_token', aggregate='mean')
        lat = results.get_metric('end_to_end_latency', aggregate='mean')
        mem = results.get_metric('memory_peak_mb', aggregate='max')
        print(f"{label:<20} {tps:<12.1f} {ttft:<12.3f} {lat:<14.3f} {mem:<14.0f}")

    # --- Benchmark with different configurations ---
    print("\n" + "=" * 60)
    print("Configuration Sweep")
    print("=" * 60)

    sweep_configs = [
        {"dtype": "float16", "flash_attention": True},
        {"dtype": "float16", "flash_attention": False},
        {"dtype": "int8", "flash_attention": True},
        {"dtype": "int4", "flash_attention": True},
    ]

    for sweep_config in sweep_configs:
        label = f"dtype={sweep_config['dtype']}_flash={'on' if sweep_config['flash_attention'] else 'off'}"
        engine = InferenceEngine(
            model_name="nexus-7b-chat",
            device="auto",
            **sweep_config,
        )
        results = runner.run(engine, model_label=label)
        tps = results.get_metric('tokens_per_second', aggregate='mean')
        print(f"  {label}: {tps:.1f} tokens/s")

    # --- Save results ---
    comparison.save_json("./benchmark_results/comparison.json")
    comparison.save_csv("./benchmark_results/comparison.csv")
    comparison.plot("./benchmark_results/comparison_charts.png")
    print("\nResults saved to ./benchmark_results/")


if __name__ == "__main__":
    main()
