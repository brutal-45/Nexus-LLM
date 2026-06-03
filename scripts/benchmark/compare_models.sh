#!/bin/bash
# Compare Models - Nexus-LLM
# ============================
# Benchmarks and compares multiple models side by side.
#
# Usage:
#   ./compare_models.sh [OPTIONS]
#
# Options:
#   -m, --models MODEL1,MODEL2,...  Comma-separated model names
#   -b, --benchmark TYPE            Benchmark type: mmlu,humaneval,throughput (default: throughput)
#   -o, --output DIR                Output directory (default: ./benchmark_results/compare)
#   -h, --help                      Show this help message

set -euo pipefail

# Default values
MODELS="nexus-3b-chat,nexus-7b-chat,nexus-13b-chat"
BENCHMARK="throughput"
OUTPUT_DIR="./benchmark_results/compare"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -b|--benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -15 "$0" | tail -13
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Nexus-LLM Model Comparison"
echo "=============================================="
echo "Models:     ${MODELS}"
echo "Benchmark:  ${BENCHMARK}"
echo "Output:     ${OUTPUT_DIR}"
echo "=============================================="
echo ""

# Convert comma-separated list to array
IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"

# System information
echo "System Information:"
echo "  Date: $(date)"
echo "  Host: $(hostname)"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
fi
echo ""

# Run benchmarks for each model
declare -A RESULTS

for MODEL in "${MODEL_ARRAY[@]}"; do
    echo "----------------------------------------------"
    echo "Benchmarking: ${MODEL}"
    echo "----------------------------------------------"

    RESULT_FILE="${OUTPUT_DIR}/${MODEL}_${TIMESTAMP}.json"

    case "${BENCHMARK}" in
        throughput)
            python -m nexus_llm.benchmark.run \
                --model "${MODEL}" \
                --device auto \
                --iterations 5 \
                --warmup 2 \
                --batch-sizes "1,8" \
                --seq-lengths "128,512,2048" \
                --output "${RESULT_FILE}"
            ;;
        mmlu)
            python -m nexus_llm.evaluation.run \
                --model "${MODEL}" \
                --benchmark mmlu \
                --output "${RESULT_FILE}"
            ;;
        humaneval)
            python -m nexus_llm.evaluation.run \
                --model "${MODEL}" \
                --benchmark humaneval \
                --output "${RESULT_FILE}"
            ;;
        *)
            echo "Unknown benchmark type: ${BENCHMARK}"
            exit 1
            ;;
    esac

    echo ""
done

# Generate comparison report
echo "=============================================="
echo "Comparison Summary"
echo "=============================================="

COMPARISON_FILE="${OUTPUT_DIR}/comparison_${TIMESTAMP}.json"

python -c "
import json
import os

models = '${MODELS}'.split(',')
timestamp = '${TIMESTAMP}'
output_dir = '${OUTPUT_DIR}'
benchmark = '${BENCHMARK}'

comparison = {}
for model in models:
    result_file = os.path.join(output_dir, f'{model}_{timestamp}.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            comparison[model] = json.load(f)

# Save combined results
comparison_file = os.path.join(output_dir, f'comparison_{timestamp}.json')
with open(comparison_file, 'w') as f:
    json.dump(comparison, f, indent=2)

# Print summary table
if benchmark == 'throughput':
    print(f\"{'Model':<20} {'Tokens/s':<12} {'TTFT (s)':<12} {'Latency (s)':<14} {'Memory (MB)':<14}\")
    print('-' * 72)
    for model, data in comparison.items():
        summary = data.get('summary', {})
        tps = summary.get('tokens_per_second', 0)
        ttft = summary.get('time_to_first_token', 0)
        lat = summary.get('end_to_end_latency', 0)
        mem = summary.get('memory_peak_mb', 0)
        print(f'{model:<20} {tps:<12.1f} {ttft:<12.3f} {lat:<14.3f} {mem:<14.0f}')
elif benchmark == 'mmlu':
    print(f\"{'Model':<20} {'Overall':<10} {'STEM':<10} {'Humanities':<12}\")
    print('-' * 52)
    for model, data in comparison.items():
        overall = data.get('overall_score', 0)
        subsets = data.get('subset_scores', {})
        print(f'{model:<20} {overall:<10.3f} {subsets.get(\"stem\", 0):<10.3f} {subsets.get(\"humanities\", 0):<12.3f}')
elif benchmark == 'humaneval':
    print(f\"{'Model':<20} {'pass@1':<10} {'pass@10':<10} {'pass@100':<10}\")
    print('-' * 50)
    for model, data in comparison.items():
        scores = data.get('scores', {})
        print(f'{model:<20} {scores.get(\"pass@1\", 0):<10.3f} {scores.get(\"pass@10\", 0):<10.3f} {scores.get(\"pass@100\", 0):<10.3f}')

print(f'\nDetailed results saved to: {comparison_file}')
"

echo ""
echo "Comparison complete!"
