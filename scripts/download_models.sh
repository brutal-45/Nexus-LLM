#!/usr/bin/env bash 
# =============================================================================
# Nexus-LLM Model Download Script
# =============================================================================
# Downloads model weights from Hugging Face Hub.
#
# Usage:
#   ./scripts/download_models.sh                                     # Download default model
#   ./scripts/download_models.sh --model meta-llama/Llama-3.1-70B    # Specific model
#   ./scripts/download_models.sh --list                              # List recommended models
#   ./scripts/download_models.sh --all-recommended                   # Download all recommended models
#   ./scripts/download_models.sh --quantize awq                      # Download quantized version
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
MODEL=""
OUTPUT_DIR="${PROJECT_ROOT}/models"
QUANTIZE="none"            # none, gptq, awq, ggml
GGML_PRECISION="Q4_K_M"   # For GGML quantization variants
LIST_ONLY=0
ALL_RECOMMENDED=0
DRY_RUN=0
HF_TOKEN="${HF_TOKEN:-}"

# Recommended models
RECOMMENDED_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "google/gemma-2-9b-it"
    "microsoft/Phi-3-medium-128k-instruct"
    "deepseek-ai/DeepSeek-V2.5"
    "01-ai/Yi-1.5-34B-Chat"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL="$2"; shift 2 ;;
        --output)         OUTPUT_DIR="$2"; shift 2 ;;
        --quantize)       QUANTIZE="$2"; shift 2 ;;
        --ggml-precision) GGML_PRECISION="$2"; shift 2 ;;
        --list)           LIST_ONLY=1; shift ;;
        --all-recommended) ALL_RECOMMENDED=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --hf-token)       HF_TOKEN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          HuggingFace model ID to download"
            echo "  --output DIR           Output directory (default: ./models)"
            echo "  --quantize METHOD      Download quantized: none, gptq, awq, ggml"
            echo "  --ggml-precision PREC  GGML precision: Q4_K_M, Q5_K_M, Q8_0 (default: Q4_K_M)"
            echo "  --list                 List recommended models"
            echo "  --all-recommended      Download all recommended models"
            echo "  --dry-run              Show what would be downloaded"
            echo "  --hf-token TOKEN       Hugging Face API token"
            echo "  -h, --help             Show this help"
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
    # Token from env or argument
    HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
}

# List recommended models
list_models() {
    echo ""
    echo "============================================"
    echo "  Recommended Models"
    echo "============================================"
    echo ""
    for i in "${!RECOMMENDED_MODELS[@]}"; do
        echo "  $((i+1)). ${RECOMMENDED_MODELS[$i]}"
    done
    echo ""
    echo "Download with:"
    echo "  ./scripts/download_models.sh --model <MODEL_ID>"
    echo ""
}

# Check disk space
check_disk_space() {
    local model_name="$1"
    local estimated_size_gb="$2"
    
    local available_gb
    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_gb=$(df -g "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
    else
        available_gb=$(df -BG "$OUTPUT_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
    fi
    
    if [ "$available_gb" -lt "$estimated_size_gb" ]; then
        log_error "Insufficient disk space: ${available_gb}GB available, ~${estimated_size_gb}GB needed"
        return 1
    fi
    log_info "Disk space OK: ${available_gb}GB available, ~${estimated_size_gb}GB needed"
}

# Download a single model
download_model() {
    local model_id="$1"
    local safe_name=$(echo "$model_id" | tr '/' '_')
    local model_dir="${OUTPUT_DIR}/${safe_name}"
    
    log_step "Downloading model: ${model_id}"
    log_info "  Output: ${model_dir}"
    log_info "  Quantize: ${QUANTIZE}"
    
    if [ "$DRY_RUN" -eq 1 ]; then
        log_info "[DRY RUN] Would download: ${model_id}"
        return
    fi
    
    mkdir -p "$model_dir"
    
    # Build huggingface-cli download command
    local cmd="huggingface-cli download"
    
    if [ -n "$HF_TOKEN" ]; then
        cmd+=" --token ${HF_TOKEN}"
    fi
    
    cmd+=" ${model_id}"
    cmd+=" --local-dir ${model_dir}"
    
    # For GGML, download specific quantization file
    if [ "$QUANTIZE" == "ggml" ]; then
        local ggml_file="*.gguf"
        cmd+=" --include \"${ggml_file}\""
    fi
    
    log_info "Running: $cmd"
    eval $cmd
    
    # Verify download
    if [ -f "${model_dir}/config.json" ] || [ -f "${model_dir}/tokenizer.json" ]; then
        log_info "Model downloaded successfully: ${model_id}"
    else
        log_warn "Download may be incomplete - verify files in ${model_dir}"
    fi
}

# Estimate model size
estimate_size() {
    local model_id="$1"
    local size_gb=10  # Default estimate
    
    if [[ "$model_id" == *"70B"* ]] || [[ "$model_id" == *"72B"* ]]; then
        size_gb=140
    elif [[ "$model_id" == *"34B"* ]] || [[ "$model_id" == *"35B"* ]]; then
        size_gb=70
    elif [[ "$model_id" == *"13B"* ]] || [[ "$model_id" == *"14B*" ]]; then
        size_gb=26
    elif [[ "$model_id" == *"8B"* ]] || [[ "$model_id" == *"7B"* ]] || [[ "$model_id" == *"9B"* ]]; then
        size_gb=16
    elif [[ "$model_id" == *"3B"* ]]; then
        size_gb=6
    elif [[ "$model_id" == *"1.5B"* ]]; then
        size_gb=3
    elif [[ "$model_id" == *"0.5B"* ]] || [[ "$model_id" == *"1B"* ]]; then
        size_gb=2
    fi
    
    # Quantized models are smaller
    if [ "$QUANTIZE" == "awq" ] || [ "$QUANTIZE" == "gptq" ]; then
        size_gb=$((size_gb / 3))
    elif [ "$QUANTIZE" == "ggml" ]; then
        size_gb=$((size_gb / 2))
    fi
    
    echo "$size_gb"
}

# Main
main() {
    load_env
    
    if [ "$LIST_ONLY" -eq 1 ]; then
        list_models
        exit 0
    fi
    
    echo ""
    echo "============================================"
    echo "  Nexus-LLM Model Downloader"
    echo "============================================"
    echo ""
    
    # Check for huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        log_error "huggingface-cli not found. Install with: pip install huggingface_hub"
        exit 1
    fi
    
    # Login if token provided
    if [ -n "$HF_TOKEN" ]; then
        log_info "Logging in to Hugging Face Hub"
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    if [ "$ALL_RECOMMENDED" -eq 1 ]; then
        log_info "Downloading all recommended models"
        for model in "${RECOMMENDED_MODELS[@]}"; do
            size=$(estimate_size "$model")
            check_disk_space "$model" "$size" && download_model "$model"
            echo ""
        done
    elif [ -n "$MODEL" ]; then
        size=$(estimate_size "$MODEL")
        check_disk_space "$MODEL" "$size"
        download_model "$MODEL"
    else
        # Use default model
        DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
        log_warn "No model specified. Downloading default: ${DEFAULT_MODEL}"
        log_info "Use --model to specify a different model, or --list to see recommendations"
        size=$(estimate_size "$DEFAULT_MODEL")
        check_disk_space "$DEFAULT_MODEL" "$size"
        download_model "$DEFAULT_MODEL"
    fi
    
    echo ""
    log_info "Download complete!"
    log_info "Models saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To use a downloaded model:"
    echo "  nexus-llm serve --model ${OUTPUT_DIR}/meta-llama_Llama-3.1-8B-Instruct"
}

main
