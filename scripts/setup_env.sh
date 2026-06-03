#!/usr/bin/env bash
# =============================================================================
# Nexus-LLM Environment Setup Script
# =============================================================================
# Sets up the complete development environment including:
# - Python virtual environment
# - System dependencies
# - NVIDIA CUDA toolkit verification
# - Pre-commit hooks
# - Configuration files
#
# Usage:
#   ./scripts/setup_env.sh                    # Full setup
#   ./scripts/setup_env.sh --skip-system      # Skip system packages
#   ./scripts/setup_env.sh --skip-cuda        # Skip CUDA check
#   ./scripts/setup_env.sh --python 3.11      # Specific Python version
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
PYTHON_VERSION="3.11"
SKIP_SYSTEM=0
SKIP_CUDA=0
SKIP_VENV=0
DEV_MODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)      PYTHON_VERSION="$2"; shift 2 ;;
        --skip-system) SKIP_SYSTEM=1; shift ;;
        --skip-cuda)   SKIP_CUDA=1; shift ;;
        --skip-venv)   SKIP_VENV=1; shift ;;
        --dev)         DEV_MODE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --python VER      Python version (default: 3.11)"
            echo "  --skip-system     Skip system package installation"
            echo "  --skip-cuda       Skip CUDA verification"
            echo "  --skip-venv       Skip virtual environment creation"
            echo "  --dev             Install development dependencies"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "  Nexus-LLM Environment Setup"
echo "================================================"
echo ""

# Install system packages
install_system_packages() {
    if [ "$SKIP_SYSTEM" -eq 1 ]; then
        log_info "Skipping system package installation"
        return
    fi

    log_step "Installing system dependencies"

    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            python3 \
            python3-pip \
            python3-venv \
            git \
            git-lfs \
            curl \
            wget \
            build-essential \
            libssl-dev \
            libffi-dev \
            libpq-dev \
            pkg-config
        log_info "System packages installed (apt)"
    elif command -v yum &> /dev/null; then
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            python3 \
            python3-pip \
            git \
            git-lfs \
            curl \
            wget \
            openssl-devel \
            libffi-devel \
            postgresql-devel
        log_info "System packages installed (yum)"
    elif command -v brew &> /dev/null; then
        brew install python@${PYTHON_VERSION} git git-lfs
        log_info "System packages installed (brew)"
    else
        log_warn "Unsupported package manager. Please install system dependencies manually."
    fi
}

# Verify CUDA
verify_cuda() {
    if [ "$SKIP_CUDA" -eq 1 ]; then
        log_info "Skipping CUDA verification"
        return
    fi

    log_step "Verifying CUDA installation"

    if command -v nvidia-smi &> /dev/null; then
        NVIDIA_VERSION=$(nvidia-smi | head -3 | tail -1 | awk '{print $6}')
        CUDA_VERSION=$(nvidia-smi | head -3 | tail -1 | awk '{print $9}')
        GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

        log_info "NVIDIA Driver: ${NVIDIA_VERSION}"
        log_info "CUDA Version:   ${CUDA_VERSION}"
        log_info "GPU:            ${GPU_NAME} (${GPU_MEMORY})"
        log_info "GPU Count:      ${GPU_COUNT}"

        if command -v nvcc &> /dev/null; then
            NVCC_VERSION=$(nvcc --version | tail -1 | awk '{print $6}')
            log_info "NVCC Version:   ${NVCC_VERSION}"
        else
            log_warn "nvcc not found. CUDA toolkit may not be installed."
            log_warn "Install from: https://developer.nvidia.com/cuda-downloads"
        fi
    else
        log_warn "nvidia-smi not found. No NVIDIA GPU detected."
        log_warn "GPU training will not be available."
    fi
}

# Create virtual environment
create_venv() {
    if [ "$SKIP_VENV" -eq 1 ]; then
        log_info "Skipping virtual environment creation"
        return
    fi

    log_step "Creating Python virtual environment"

    VENV_DIR="${PROJECT_ROOT}/.venv"
    PYTHON_CMD="python${PYTHON_VERSION}"

    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        PYTHON_CMD="python3"
    fi

    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment exists at $VENV_DIR"
        read -rp "Recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            source "$VENV_DIR/bin/activate"
            log_info "Using existing virtual environment"
            return
        fi
    fi

    $PYTHON_CMD -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    log_info "Virtual environment created with $($PYTHON_CMD --version)"
}

# Install Python dependencies
install_python_deps() {
    log_step "Installing Python dependencies"

    cd "$PROJECT_ROOT"

    if [ "$DEV_MODE" -eq 1 ]; then
        pip install -e ".[dev,test,lint,typecheck]"
        log_info "Development dependencies installed"
    else
        pip install -e "."
        log_info "Core dependencies installed"
    fi
}

# Setup pre-commit
setup_precommit() {
    log_step "Setting up pre-commit hooks"

    if command -v pre-commit &> /dev/null; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_info "Pre-commit hooks installed"
    else
        log_warn "pre-commit not found. Install with: pip install pre-commit"
    fi
}

# Setup configuration
setup_config() {
    log_step "Setting up configuration files"

    # .env file
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
        log_info "Created .env from .env.example"
        log_warn "Please edit .env with your settings!"
    else
        log_info ".env already exists"
    fi

    # Create directories
    mkdir -p "${PROJECT_ROOT}/models"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data/vector_store"
    mkdir -p "${PROJECT_ROOT}/checkpoints"
    mkdir -p "${PROJECT_ROOT}/benchmarks"
    log_info "Project directories created"
}

# Verify setup
verify_setup() {
    log_step "Verifying setup"

    python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import nexus_llm
    print(f'Nexus-LLM: {nexus_llm.__version__}')
except ImportError:
    print('WARNING: nexus_llm import failed')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
except ImportError:
    print('WARNING: PyTorch not installed (CPU-only mode)')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('WARNING: transformers not installed')
"
}

# Print completion message
print_completion() {
    echo ""
    echo "================================================"
    log_info "Environment setup complete!"
    echo "================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Activate:  source .venv/bin/activate"
    echo "  2. Configure: edit .env with your settings"
    echo "  3. Run:       nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct"
    echo ""
}

# Main
main() {
    install_system_packages
    verify_cuda
    create_venv
    install_python_deps
    setup_precommit
    setup_config
    verify_setup
    print_completion
}

main
