#!/usr/bin/env bash
# ============================================================================
# Nexus-LLM Installation Script
# ============================================================================
# This script sets up the Nexus-LLM environment:
#   - Checks Python version (3.9+)
#   - Creates a virtual environment
#   - Installs dependencies
#   - Downloads the default model (gpt2-medium)
#   - Creates necessary directories
#
# Usage:
#   chmod +x scripts/install.sh
#   ./scripts/install.sh
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color definitions
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
banner()  { echo -e "\n${BOLD}${CYAN}========================================${NC}"; \
            echo -e "${BOLD}${CYAN}  $*${NC}"; \
            echo -e "${BOLD}${CYAN}========================================${NC}\n"; }

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
DEFAULT_MODEL="gpt2-medium"

# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------
cleanup() {
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        error "Installation failed with exit code ${exit_code}."
        echo -e "  ${DIM}Check the output above for details.${NC}"
        echo -e "  ${DIM}You can re-run this script to retry.${NC}"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 1: Check Python version
# ---------------------------------------------------------------------------
banner "Nexus-LLM Installer v2.0.0"

info "Checking Python version..."

if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed or not in PATH."
    echo -e "  ${DIM}Please install Python 3.9 or later: https://www.python.org/downloads/${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    error "Python 3.9+ is required, but found Python ${PYTHON_VERSION}."
    echo -e "  ${DIM}Please upgrade: https://www.python.org/downloads/${NC}"
    exit 1
fi

success "Python ${PYTHON_VERSION} detected."

# ---------------------------------------------------------------------------
# Step 2: Create virtual environment
# ---------------------------------------------------------------------------
info "Creating virtual environment at ${VENV_DIR}..."

if [ -d "${VENV_DIR}" ]; then
    warn "Virtual environment already exists. Removing and recreating..."
    rm -rf "${VENV_DIR}"
fi

python3 -m venv "${VENV_DIR}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    error "Failed to create virtual environment."
    exit 1
fi

success "Virtual environment created."

# Activate the venv for the rest of the script
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install wheel
info "Upgrading pip and installing build tools..."
python3 -m pip install --upgrade pip setuptools wheel --quiet
success "Build tools updated."

# ---------------------------------------------------------------------------
# Step 3: Install dependencies
# ---------------------------------------------------------------------------
info "Installing Nexus-LLM dependencies..."
echo -e "  ${DIM}This may take a few minutes on first run...${NC}"

if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    python3 -m pip install -e "${PROJECT_ROOT}" 2>&1 | while IFS= read -r line; do
        # Only show key lines to avoid flooding the terminal
        if echo "$line" | grep -qiE "successfully|error|failed|requirement"; then
            echo "  $line"
        fi
    done
elif [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    python3 -m pip install -r "${PROJECT_ROOT}/requirements.txt" 2>&1 | while IFS= read -r line; do
        if echo "$line" | grep -qiE "successfully|error|failed|requirement"; then
            echo "  $line"
        fi
    done
else
    error "No pyproject.toml or requirements.txt found."
    exit 1
fi

success "Dependencies installed."

# ---------------------------------------------------------------------------
# Step 4: Create necessary directories
# ---------------------------------------------------------------------------
info "Creating project directories..."

DIRECTORIES=(
    "${PROJECT_ROOT}/models"
    "${PROJECT_ROOT}/data"
    "${PROJECT_ROOT}/logs"
    "${PROJECT_ROOT}/config"
    "${PROJECT_ROOT}/checkpoints"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "  ${GREEN}created${NC}  ${dir}"
    else
        echo -e "  ${DIM}exists${NC}   ${dir}"
    fi
done

success "Directories ready."

# ---------------------------------------------------------------------------
# Step 5: Download the default model
# ---------------------------------------------------------------------------
info "Downloading default model (${DEFAULT_MODEL})..."
echo -e "  ${DIM}This downloads ~1.5 GB from HuggingFace Hub.${NC}"
echo -e "  ${DIM}You can skip this with Ctrl+C and download later.${NC}"

if python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('openai-community/${DEFAULT_MODEL}')" 2>/dev/null; then
    success "Default model '${DEFAULT_MODEL}' downloaded and cached."
else
    warn "Model download failed or was skipped."
    echo -e "  ${DIM}You can download it later with:${NC}"
    echo -e "  ${CYAN}  python scripts/download_model.py ${DEFAULT_MODEL}${NC}"
fi

# ---------------------------------------------------------------------------
# Step 6: Deactivate venv
# ---------------------------------------------------------------------------
deactivate || true

# ---------------------------------------------------------------------------
# Success!
# ---------------------------------------------------------------------------
banner "Installation Complete!"

echo -e "  ${BOLD}Nexus-LLM is ready to use!${NC}"
echo ""
echo -e "  ${CYAN}Next steps:${NC}"
echo -e "    1. Activate the virtual environment:"
echo -e "       ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo -e "    2. Start chatting:"
echo -e "       ${GREEN}nexus-llm chat${NC}"
echo -e "       ${GREEN}nexus-llm chat --model gpt2-medium${NC}"
echo ""
echo -e "    3. Or use the quick-start script:"
echo -e "       ${GREEN}./scripts/start.sh${NC}"
echo ""
echo -e "    4. Download more models:"
echo -e "       ${GREEN}nexus-llm download phi-2${NC}"
echo -e "       ${GREEN}nexus-llm models${NC}  ${DIM}(list all 39 models)${NC}"
echo ""
echo -e "    5. Start the API server:"
echo -e "       ${GREEN}nexus-llm serve --port 8000${NC}"
echo ""
echo -e "  ${DIM}Documentation: https://github.com/brutal-45/Nexus-LLM${NC}"
echo -e "  ${DIM}Config file:   ${PROJECT_ROOT}/config/default_config.yaml${NC}"
echo ""
