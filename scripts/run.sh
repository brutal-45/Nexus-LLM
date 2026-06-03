#!/usr/bin/env bash 
# =============================================================================
# Nexus-LLM Server Run Script
# =============================================================================
# Usage:
#   ./scripts/run.sh                                    # Default server
#   ./scripts/run.sh --model meta-llama/Llama-3.1-70B   # Specific model
#   ./scripts/run.sh --gpu 0,1,2,3 --port 8080          # Multi-GPU
#   ./scripts/run.sh --quantize awq                     # With quantization
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

# Default configuration
MODEL="${NEXUS_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
HOST="${NEXUS_HOST:-0.0.0.0}"
PORT="${NEXUS_PORT:-8000}"
WORKERS="${NEXUS_WORKERS:-4}"
GPU_IDS="${NEXUS_GPU_IDS:-0}"
QUANTIZE="${NEXUS_QUANTIZE:-none}"
MAX_MODEL_LEN="${NEXUS_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${NEXUS_GPU_MEMORY_UTILIZATION:-0.9}"
CONFIG_FILE=""
DAEMON=0
VERBOSE=0
RELOAD=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL="$2"; shift 2 ;;
        --host)          HOST="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --gpus)          GPU_IDS="$2"; shift 2 ;;
        --quantize)      QUANTIZE="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem)       GPU_MEM_UTIL="$2"; shift 2 ;;
        --config)        CONFIG_FILE="$2"; shift 2 ;;
        --daemon)        DAEMON=1; shift ;;
        --reload)        RELOAD=1; shift ;;
        --verbose)       VERBOSE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL         Model name or path (default: meta-llama/Llama-3.1-8B-Instruct)"
            echo "  --host HOST           Server host (default: 0.0.0.0)"
            echo "  --port PORT           Server port (default: 8000)"
            echo "  --workers N           Number of workers (default: 4)"
            echo "  --gpus IDS            Comma-separated GPU IDs (default: 0)"
            echo "  --quantize METHOD     Quantization: none, gptq, awq, ggml (default: none)"
            echo "  --max-model-len LEN   Maximum sequence length (default: 8192)"
            echo "  --gpu-mem FRAC        GPU memory utilization 0.0-1.0 (default: 0.9)"
            echo "  --config FILE         Path to config YAML file"
            echo "  --daemon              Run as daemon"
            echo "  --reload              Enable auto-reload (development)"
            echo "  --verbose             Verbose output"
            echo "  -h, --help            Show this help"
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
        log_info "Loaded .env file"
    fi

    # Activate virtual environment if available
    if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
    fi
}

# Prerequisites check
check_prerequisites() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi

    python3 -c "import nexus_llm" 2>/dev/null || {
        log_error "Nexus-LLM not installed. Run: pip install -e ."
        exit 1
    }

    # Check GPU availability if CUDA is requested
    if [ "$GPU_IDS" != "cpu" ]; then
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
            log_info "Detected ${GPU_COUNT} GPU(s)"
        else
            log_warn "nvidia-smi not found. Running on CPU only."
            GPU_IDS="cpu"
        fi
    fi
}

# Run the server
run_server() {
    log_info "Starting Nexus-LLM server"
    log_info "  Model:      ${MODEL}"
    log_info "  Host:       ${HOST}:${PORT}"
    log_info "  Workers:    ${WORKERS}"
    log_info "  GPUs:       ${GPU_IDS}"
    log_info "  Quantize:   ${QUANTIZE}"
    log_info "  Max length: ${MAX_MODEL_LEN}"

    # Build the command
    CMD="nexus-llm serve"
    CMD+=" --model ${MODEL}"
    CMD+=" --host ${HOST}"
    CMD+=" --port ${PORT}"
    CMD+=" --workers ${WORKERS}"
    CMD+=" --gpu-ids ${GPU_IDS}"
    CMD+=" --max-model-len ${MAX_MODEL_LEN}"
    CMD+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"

    if [ "$QUANTIZE" != "none" ]; then
        CMD+=" --quantize ${QUANTIZE}"
    fi

    if [ -n "$CONFIG_FILE" ]; then
        CMD+=" --config ${CONFIG_FILE}"
    fi

    if [ "$RELOAD" -eq 1 ]; then
        CMD+=" --reload"
    fi

    if [ "$VERBOSE" -eq 1 ]; then
        CMD+=" --verbose"
    fi

    # Export GPU visibility
    if [ "$GPU_IDS" != "cpu" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    fi

    if [ "$DAEMON" -eq 1 ]; then
        log_info "Running as daemon..."
        nohup $CMD > "${PROJECT_ROOT}/logs/server.log" 2>&1 &
        SERVER_PID=$!
        echo "$SERVER_PID" > "${PROJECT_ROOT}/logs/server.pid"
        log_info "Server PID: ${SERVER_PID}"
        log_info "Logs: ${PROJECT_ROOT}/logs/server.log"
    else
        exec $CMD
    fi
}

# Main
main() {
    mkdir -p "${PROJECT_ROOT}/logs"
    load_env
    check_prerequisites
    run_server
}

main
