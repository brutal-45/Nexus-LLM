#!/bin/bash
# Run Benchmarks - Nexus-LLM
# ============================
# Runs the standard benchmark suite against one or more models.
#
# Usage:
#   ./run_benchmark.sh [OPTIONS]
#
# Options:
#   -m, --model MODEL       Model to benchmark (default: nexus-7b-chat)
#   -d, --device DEVICE     Device to use (default: auto)
#   -i, --iterations N      Number of benchmark iterations (default: 10)
#   -w, --warmup N          Number of warmup iterations (default: 3)
#   -b, --batch-sizes SIZES Comma-separated batch sizes (default: 1,4,8,16)
#   -l, --seq-lengths LENS  Comma-separated sequence lengths (default: 128,512,2048)
#   -o, --output DIR        Output directory (default: ./benchmark_results)
#   -q, --quantization TYPE Quantization type: fp16,int8,int4 (default: fp16)
#   -v, --verbose           Enable verbose output
#   -h, --help              Show this help message

set -euo pipefail

# Default values
MODEL="nexus-7b-chat"
DEVICE="auto"
ITERATIONS=10
WARMUP=3
BATCH_SIZES="1,4,8,16"
SEQ_LENGTHS="128,512,2048"
OUTPUT_DIR="./benchmark_results"
QUANTIZATION="fp16"
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -b|--batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        -l|--seq-lengths)
            SEQ_LENGTHS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -18
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
RESULT_FILE="${OUTPUT_DIR}/benchmark_${MODEL}_${TIMESTAMP}.json"

echo "=============================================="
echo "Nexus-LLM Benchmark"
echo "=============================================="
echo "Model:          ${MODEL}"
echo "Device:         ${DEVICE}"
echo "Quantization:   ${QUANTIZATION}"
echo "Iterations:     ${ITERATIONS}"
echo "Warmup:         ${WARMUP}"
echo "Batch sizes:    ${BATCH_SIZES}"
echo "Seq lengths:    ${SEQ_LENGTHS}"
echo "Output:         ${RESULT_FILE}"
echo "=============================================="
echo ""

# Check GPU availability
if [[ "${DEVICE}" == "auto" || "${DEVICE}" == "cuda" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo ""
    else
        echo "WARNING: nvidia-smi not found. GPU may not be available."
        echo ""
    fi
fi

# Run the benchmark
echo "Starting benchmark..."
echo ""

VERBOSE_FLAG=""
if [[ "${VERBOSE}" == true ]]; then
    VERBOSE_FLAG="--verbose"
fi

python -m nexus_llm.benchmark.run \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${QUANTIZATION}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    --batch-sizes "${BATCH_SIZES}" \
    --seq-lengths "${SEQ_LENGTHS}" \
    --output "${RESULT_FILE}" \
    ${VERBOSE_FLAG}

# Check results
if [[ -f "${RESULT_FILE}" ]]; then
    echo ""
    echo "=============================================="
    echo "Benchmark Complete!"
    echo "=============================================="
    echo "Results saved to: ${RESULT_FILE}"
    echo ""
    echo "Quick Summary:"
    python -c "
import json
with open('${RESULT_FILE}') as f:
    data = json.load(f)
for model_name, results in data.items():
    print(f'  Model: {model_name}')
    for metric, value in results.get('summary', {}).items():
        if isinstance(value, float):
            print(f'    {metric}: {value:.2f}')
        else:
            print(f'    {metric}: {value}')
"
else
    echo "ERROR: Benchmark results file not found."
    exit 1
fi
