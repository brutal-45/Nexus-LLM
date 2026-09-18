#!/usr/bin/env bash
# ============================================================================
# Nexus-LLM Quick Start Script
# ============================================================================
# Activates the virtual environment and starts an interactive chat session
# with the default model (gpt2-medium).
#
# Usage:
#   chmod +x scripts/start.sh
#   ./scripts/start.sh [model_id]
#
# Examples:
#   ./scripts/start.sh              # use default model (gpt2-medium)
#   ./scripts/start.sh phi-2        # start with phi-2
#   ./scripts/start.sh tinyllama    # start with TinyLlama
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color definitions
# ---------------------------------------------------------------------------
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
DIM='\033[2m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
DEFAULT_MODEL="gpt2-medium"
MODEL="${1:-$DEFAULT_MODEL}"

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${VENV_DIR}${NC}"
    echo -e "${YELLOW}Run the install script first:${NC}"
    echo -e "  ${GREEN}./scripts/install.sh${NC}"
    exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Launch chat
# ---------------------------------------------------------------------------
echo -e "${CYAN}Starting Nexus-LLM with model: ${MODEL}${NC}"
echo -e "${DIM}Press Ctrl+C to exit.${NC}"
echo ""

nexus-llm chat --model "${MODEL}"
