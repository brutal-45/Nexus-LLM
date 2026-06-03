#!/usr/bin/env bash
# =============================================================================
# Nexus-LLM Benchmark Script
# =============================================================================
# Runs inference benchmarks measuring throughput, latency, and quality metrics.
#
# Usage:
#   ./scripts/benchmark.sh                                     # Default benchmark
#   ./scripts/benchmark.sh --model meta-llama/Llama-3.1-70B    # Specific model
#   ./scripts/benchmark.sh --quantize awq --batch-sizes 1,8,32 # With quantization
#   ./scripts/benchmark.sh --benchmark mmlu,gsm8k              # Quality benchmarks
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# Defaults
MODEL="${NEXUS_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QUANTIZE="none"
BATCH_SIZES="1,4,8,16,32"
INPUT_LENGTHS="128,512,1024,2048"
MAX_TOKENS=256
NUM_WARMUP=3
NUM_ITERATIONS=10
BENCHMARK="throughput"   # throughput, mmlu, gsm8k, humaneval, mt_bench, all
OUTPUT_DIR="./benchmarks"
GPU_IDS="0"
WARMUP_GPU=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL="$2"; shift 2 ;;
        --quantize)      QUANTIZE="$2"; shift 2 ;;
        --batch-sizes)   BATCH_SIZES="$2"; shift 2 ;;
        --input-lengths) INPUT_LENGTHS="$2"; shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
        --warmup)        NUM_WARMUP="$2"; shift 2 ;;
        --iterations)    NUM_ITERATIONS="$2"; shift 2 ;;
        --benchmark)     BENCHMARK="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --gpus)          GPU_IDS="$2"; shift 2 ;;
        --no-warmup-gpu) WARMUP_GPU=0; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model to benchmark"
            echo "  --quantize METHOD       Quantization: none, gptq, awq, ggml"
            echo "  --batch-sizes SIZES     Comma-separated batch sizes (default: 1,4,8,16,32)"
            echo "  --input-lengths LENS    Comma-separated input lengths (default: 128,512,1024,2048)"
            echo "  --max-tokens N          Max output tokens (default: 256)"
            echo "  --warmup N              Warmup iterations (default: 3)"
            echo "  --iterations N          Benchmark iterations (default: 10)"
            echo "  --benchmark TYPE        Benchmark type: throughput, mmlu, gsm8k, humaneval, mt_bench, all"
            echo "  --output DIR            Output directory (default: ./benchmarks)"
            echo "  --gpus IDS              GPU IDs (default: 0)"
            echo "  --no-warmup-gpu         Skip GPU warmup"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load environment
load_env() {
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
    if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
    fi
}

# GPU warmup
warmup_gpu() {
    if [ "$WARMUP_GPU" -eq 0 ]; then
        return
    fi
    
    log_step "Warming up GPU"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        # Allocate and compute on each GPU to stabilize clocks
        x = torch.randn(1000, 1000, device=f'cuda:{i}')
        y = torch.matmul(x, x)
        torch.cuda.synchronize(i)
    print(f'GPU warmup complete ({torch.cuda.device_count()} device(s))')
else:
    print('No GPU available, skipping warmup')
"
}

# Run throughput benchmark
run_throughput_benchmark() {
    log_step "Running throughput benchmark"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="${OUTPUT_DIR}/throughput_${TIMESTAMP}.json"
    
    CMD="nexus-llm benchmark throughput"
    CMD+=" --model ${MODEL}"
    CMD+=" --quantize ${QUANTIZE}"
    CMD+=" --batch-sizes ${BATCH_SIZES}"
    CMD+=" --input-lengths ${INPUT_LENGTHS}"
    CMD+=" --max-tokens ${MAX_TOKENS}"
    CMD+=" --warmup ${NUM_WARMUP}"
    CMD+=" --iterations ${NUM_ITERATIONS}"
    CMD+=" --output ${RESULT_FILE}"
    
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    
    log_info "Running: $CMD"
    eval $CMD
    
    log_info "Results saved to: ${RESULT_FILE}"
}

# Run quality benchmarks
run_quality_benchmark() {
    local bench_type="$1"
    log_step "Running quality benchmark: ${bench_type}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="${OUTPUT_DIR}/${bench_type}_${TIMESTAMP}.json"
    
    CMD="nexus-llm benchmark ${bench_type}"
    CMD+=" --model ${MODEL}"
    CMD+=" --quantize ${QUANTIZE}"
    CMD+=" --output ${RESULT_FILE}"
    
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    
    log_info "Running: $CMD"
    eval $CMD
    
    log_info "Results saved to: ${RESULT_FILE}"
}

# Generate summary report
generate_report() {
    log_step "Generating summary report"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    REPORT_FILE="${OUTPUT_DIR}/report_${TIMESTAMP}.md"
    
    python3 -c "
import json, glob, os

output_dir = '${OUTPUT_DIR}'
report_lines = []
report_lines.append('# Nexus-LLM Benchmark Report')
report_lines.append(f'Model: ${MODEL}')
report_lines.append(f'Quantization: ${QUANTIZE}')
report_lines.append(f'Date: $(date -Iseconds)')
report_lines.append('')

# Collect all result files
for result_file in sorted(glob.glob(os.path.join(output_dir, '*.json'))):
    try:
        with open(result_file) as f:
            data = json.load(f)
        report_lines.append(f'## {os.path.basename(result_file)}')
        report_lines.append('')
        report_lines.append('\`\`\`json')
        report_lines.append(json.dumps(data, indent=2)[:2000])
        report_lines.append('\`\`\`')
        report_lines.append('')
    except Exception as e:
        report_lines.append(f'Error reading {result_file}: {e}')

with open('${REPORT_FILE}', 'w') as f:
    f.write('\n'.join(report_lines))

print(f'Report saved to: ${REPORT_FILE}')
"
}

# Main
main() {
    load_env
    mkdir -p "$OUTPUT_DIR"
    warmup_gpu
    
    echo ""
    echo "============================================"
    echo "  Nexus-LLM Benchmark"
    echo "============================================"
    echo "  Model:        ${MODEL}"
    echo "  Quantize:     ${QUANTIZE}"
    echo "  Benchmark:    ${BENCHMARK}"
    echo "  Batch sizes:  ${BATCH_SIZES}"
    echo "  GPUs:         ${GPU_IDS}"
    echo "============================================"
    echo ""
    
    case "$BENCHMARK" in
        throughput)
            run_throughput_benchmark
            ;;
        mmlu|gsm8k|humaneval|mt_bench)
            run_quality_benchmark "$BENCHMARK"
            ;;
        all)
            run_throughput_benchmark
            for bench in mmlu gsm8k humaneval mt_bench; do
                run_quality_benchmark "$bench"
            done
            ;;
        *)
            log_error "Unknown benchmark type: $BENCHMARK"
            exit 1
            ;;
    esac
    
    generate_report
    log_info "Benchmark complete!"
}

main
