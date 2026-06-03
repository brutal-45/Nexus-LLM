#!/usr/bin/env bash 
# =============================================================================
# Nexus-LLM Training Script
# =============================================================================
# Usage:
#   ./scripts/train.sh --data ./data/train.jsonl                  # LoRA (default)
#   ./scripts/train.sh --data ./data/train.jsonl --method full    # Full fine-tune
#   ./scripts/train.sh --data ./data/train.jsonl --method dpo     # DPO alignment
#   ./scripts/train.sh --data ./data/train.jsonl --deepspeed z3   # DeepSpeed ZeRO-3
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

# Default training configuration
MODEL="${NEXUS_MODEL:-meta-llama/Llama-3.1-8B}"
DATA_PATH=""
METHOD="lora"               # lora, qlora, full, dpo, orpo, rlhf
OUTPUT_DIR="./checkpoints"
EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=8
LEARNING_RATE="2e-4"
MAX_SEQ_LENGTH=2048
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
LOGGING_STEPS=10
SAVE_STEPS=500
EVAL_STEPS=500
EVAL_STRATEGY="steps"
FP16=0
BF16=1
DEEPSPEED=""
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
SEED=42
WANDB=0
RESUME_FROM=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)               MODEL="$2"; shift 2 ;;
        --data)                DATA_PATH="$2"; shift 2 ;;
        --method)              METHOD="$2"; shift 2 ;;
        --output)              OUTPUT_DIR="$2"; shift 2 ;;
        --epochs)              EPOCHS="$2"; shift 2 ;;
        --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)          GRADIENT_ACCUMULATION="$2"; shift 2 ;;
        --lr)                  LEARNING_RATE="$2"; shift 2 ;;
        --max-seq-length)      MAX_SEQ_LENGTH="$2"; shift 2 ;;
        --warmup-ratio)        WARMUP_RATIO="$2"; shift 2 ;;
        --weight-decay)        WEIGHT_DECAY="$2"; shift 2 ;;
        --logging-steps)       LOGGING_STEPS="$2"; shift 2 ;;
        --save-steps)          SAVE_STEPS="$2"; shift 2 ;;
        --eval-steps)          EVAL_STEPS="$2"; shift 2 ;;
        --deepspeed)           DEEPSPEED="$2"; shift 2 ;;
        --lora-rank)           LORA_RANK="$2"; shift 2 ;;
        --lora-alpha)          LORA_ALPHA="$2"; shift 2 ;;
        --lora-dropout)        LORA_DROPOUT="$2"; shift 2 ;;
        --lora-target)         LORA_TARGET="$2"; shift 2 ;;
        --fp16)                FP16=1; BF16=0; shift ;;
        --bf16)                BF16=1; FP16=0; shift ;;
        --wandb)               WANDB=1; shift ;;
        --seed)                SEED="$2"; shift 2 ;;
        --resume)              RESUME_FROM="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --data DATA_PATH [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --data PATH            Training data path (JSONL format)"
            echo ""
            echo "Model:"
            echo "  --model MODEL          Base model (default: meta-llama/Llama-3.1-8B)"
            echo "  --method METHOD        Training method: lora, qlora, full, dpo, orpo, rlhf (default: lora)"
            echo ""
            echo "Training:"
            echo "  --output DIR           Output directory (default: ./checkpoints)"
            echo "  --epochs N             Number of epochs (default: 3)"
            echo "  --batch-size N         Batch size per GPU (default: 4)"
            echo "  --grad-accum N         Gradient accumulation steps (default: 8)"
            echo "  --lr RATE              Learning rate (default: 2e-4)"
            echo "  --max-seq-length N     Maximum sequence length (default: 2048)"
            echo "  --warmup-ratio RATIO   Warmup ratio (default: 0.1)"
            echo "  --weight-decay DECAY   Weight decay (default: 0.01)"
            echo "  --seed N               Random seed (default: 42)"
            echo ""
            echo "LoRA:"
            echo "  --lora-rank N          LoRA rank (default: 16)"
            echo "  --lora-alpha N         LoRA alpha (default: 32)"
            echo "  --lora-dropout DROP    LoRA dropout (default: 0.05)"
            echo "  --lora-target MODULES  Target modules (default: all linear)"
            echo ""
            echo "Distributed:"
            echo "  --deepspeed CONFIG     DeepSpeed config: z1, z2, z3, or path"
            echo ""
            echo "Other:"
            echo "  --fp16                 Use FP16 precision"
            echo "  --bf16                 Use BF16 precision (default)"
            echo "  --wandb                Enable Weights & Biases logging"
            echo "  --resume PATH          Resume from checkpoint"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_PATH" ]; then
    log_error "Training data path is required. Use --data PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    log_error "Training data file not found: $DATA_PATH"
    exit 1
fi

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

# Setup DeepSpeed
setup_deepspeed() {
    if [ -z "$DEEPSPEED" ]; then
        return
    fi

    local ds_config="${PROJECT_ROOT}/configs/deepspeed"
    case "$DEEPSPEED" in
        z1) DEEPSPEED="${ds_config}/zero1.json" ;;
        z2) DEEPSPEED="${ds_config}/zero2.json" ;;
        z3) DEEPSPEED="${ds_config}/zero3.json" ;;
    esac

    if [ ! -f "$DEEPSPEED" ]; then
        log_warn "DeepSpeed config not found: $DEEPSPEED. Generating default..."
        mkdir -p "$(dirname "$DEEPSPEED")"
        python3 -c "
import json
config = {
    'bf16': {'enabled': True},
    'zero_optimization': {
        'stage': ${DEEPSPEED##*zero},
        'overlap_comm': True,
        'contiguous_gradients': True,
    },
    'gradient_accumulation_steps': ${GRADIENT_ACCUMULATION},
    'train_micro_batch_size_per_gpu': ${BATCH_SIZE},
    'gradient_clipping': 1.0,
}
with open('${DEEPSPEED}', 'w') as f:
    json.dump(config, f, indent=2)
"
    fi
}

# Setup W&B
setup_wandb() {
    if [ "$WANDB" -eq 1 ]; then
        export WANDB_PROJECT="${WANDB_PROJECT:-nexus-llm}"
        if [ -n "${WANDB_API_KEY:-}" ]; then
            wandb login "$WANDB_API_KEY" 2>/dev/null || true
        fi
        log_info "Weights & Biases enabled (project: $WANDB_PROJECT)"
    fi
}

# Print training summary
print_summary() {
    echo ""
    echo "============================================"
    echo "  Nexus-LLM Training Configuration"
    echo "============================================"
    echo "  Model:              ${MODEL}"
    echo "  Method:             ${METHOD}"
    echo "  Data:               ${DATA_PATH}"
    echo "  Output:             ${OUTPUT_DIR}"
    echo "  Epochs:             ${EPOCHS}"
    echo "  Batch size:         ${BATCH_SIZE}"
    echo "  Gradient accum:     ${GRADIENT_ACCUMULATION}"
    echo "  Effective batch:    $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
    echo "  Learning rate:      ${LEARNING_RATE}"
    echo "  Max sequence:       ${MAX_SEQ_LENGTH}"
    echo "  Precision:          $([ "$BF16" -eq 1 ] && echo "BF16" || echo "FP16")"
    if [[ "$METHOD" == lora || "$METHOD" == qlora ]]; then
        echo "  LoRA rank:          ${LORA_RANK}"
        echo "  LoRA alpha:         ${LORA_ALPHA}"
    fi
    if [ -n "$DEEPSPEED" ]; then
        echo "  DeepSpeed:          ${DEEPSPEED}"
    fi
    echo "============================================"
    echo ""
}

# Execute training
run_training() {
    mkdir -p "$OUTPUT_DIR"
    
    CMD="nexus-llm train"
    CMD+=" --model ${MODEL}"
    CMD+=" --data ${DATA_PATH}"
    CMD+=" --method ${METHOD}"
    CMD+=" --output-dir ${OUTPUT_DIR}"
    CMD+=" --epochs ${EPOCHS}"
    CMD+=" --batch-size ${BATCH_SIZE}"
    CMD+=" --gradient-accumulation-steps ${GRADIENT_ACCUMULATION}"
    CMD+=" --learning-rate ${LEARNING_RATE}"
    CMD+=" --max-seq-length ${MAX_SEQ_LENGTH}"
    CMD+=" --warmup-ratio ${WARMUP_RATIO}"
    CMD+=" --weight-decay ${WEIGHT_DECAY}"
    CMD+=" --logging-steps ${LOGGING_STEPS}"
    CMD+=" --save-steps ${SAVE_STEPS}"
    CMD+=" --eval-steps ${EVAL_STEPS}"
    CMD+=" --eval-strategy ${EVAL_STRATEGY}"
    CMD+=" --seed ${SEED}"

    # Precision
    if [ "$BF16" -eq 1 ]; then
        CMD+=" --bf16"
    elif [ "$FP16" -eq 1 ]; then
        CMD+=" --fp16"
    fi

    # LoRA config
    if [[ "$METHOD" == lora || "$METHOD" == qlora ]]; then
        CMD+=" --lora-rank ${LORA_RANK}"
        CMD+=" --lora-alpha ${LORA_ALPHA}"
        CMD+=" --lora-dropout ${LORA_DROPOUT}"
        CMD+=" --lora-target-modules ${LORA_TARGET}"
    fi

    # DeepSpeed
    if [ -n "$DEEPSPEED" ]; then
        CMD+=" --deepspeed ${DEEPSPEED}"
    fi

    # W&B
    if [ "$WANDB" -eq 1 ]; then
        CMD+=" --report-to wandb"
    fi

    # Resume
    if [ -n "$RESUME_FROM" ]; then
        CMD+=" --resume-from-checkpoint ${RESUME_FROM}"
    fi

    log_info "Starting training..."
    exec $CMD
}

# Main
main() {
    load_env
    setup_deepspeed
    setup_wandb
    print_summary
    run_training
}

main
