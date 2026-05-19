#!/usr/bin/env bash
# =============================================================================
# Nexus-LLM Installation Script
# =============================================================================
# Usage:
#   ./scripts/install.sh              # Basic installation
#   ./scripts/install.sh --gpu         # With GPU support
#   ./scripts/install.sh --dev         # Development dependencies
#   ./scripts/install.sh --all         # All extras
#   ./scripts/install.sh --version 3.11  # Specific Python version
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
INSTALL_MODE="basic"
PYTHON_VERSION=""
VERBOSE=0
SKIP_VENV=0
UPGRADE_PIP=1

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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)       INSTALL_MODE="gpu"; shift ;;
        --dev)       INSTALL_MODE="dev"; shift ;;
        --all)       INSTALL_MODE="all"; shift ;;
        --basic)     INSTALL_MODE="basic"; shift ;;
        --version)   PYTHON_VERSION="$2"; shift 2 ;;
        --skip-venv) SKIP_VENV=1; shift ;;
        --no-upgrade) UPGRADE_PIP=0; shift ;;
        --verbose)   VERBOSE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu          Install with GPU support"
            echo "  --dev          Install development dependencies"
            echo "  --all          Install all extras"
            echo "  --basic        Basic installation (default)"
            echo "  --version VER  Specify Python version"
            echo "  --skip-venv    Skip virtual environment creation"
            echo "  --no-upgrade   Don't upgrade pip"
            echo "  --verbose      Verbose output"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "  Nexus-LLM Installer"
echo "  Mode: ${INSTALL_MODE}"
echo "================================================"
echo ""

# Check Python
check_python() {
    log_step "Checking Python installation"
    
    if [ -n "$PYTHON_VERSION" ]; then
        PYTHON_CMD="python${PYTHON_VERSION}"
    else
        PYTHON_CMD="python3"
    fi
    
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        log_error "Python not found: $PYTHON_CMD"
        log_info "Please install Python 3.9+ from https://www.python.org/"
        exit 1
    fi
    
    PY_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Found Python: $PY_VERSION"
    
    # Check minimum version
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
        log_error "Python 3.9+ is required (found $PY_VERSION)"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    if [ "$SKIP_VENV" -eq 1 ]; then
        log_info "Skipping virtual environment creation"
        return
    fi
    
    log_step "Creating virtual environment"
    
    VENV_DIR="${PROJECT_ROOT}/.venv"
    
    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment already exists at $VENV_DIR"
        read -rp "Remove and recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            source "$VENV_DIR/bin/activate"
            return
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    log_info "Virtual environment created at $VENV_DIR"
}

# Install dependencies
install_deps() {
    log_step "Installing Nexus-LLM (${INSTALL_MODE})"
    
    cd "$PROJECT_ROOT"
    
    if [ "$UPGRADE_PIP" -eq 1 ]; then
        pip install --upgrade pip setuptools wheel
    fi
    
    case "$INSTALL_MODE" in
        basic)
            pip install -e "."
            ;;
        gpu)
            pip install -e ".[gpu]"
            ;;
        dev)
            pip install -e ".[dev]"
            ;;
        all)
            pip install -e ".[all]"
            ;;
    esac
    
    log_info "Installation complete"
}

# Verify installation
verify_installation() {
    log_step "Verifying installation"
    
    python -c "
import nexus_llm
print(f'Nexus-LLM version: {nexus_llm.__version__}')
print('Installation verified successfully!')
" || {
        log_error "Installation verification failed"
        exit 1
    }
}

# Install pre-commit hooks
install_precommit() {
    if [ "$INSTALL_MODE" = "dev" ] || [ "$INSTALL_MODE" = "all" ]; then
        log_step "Installing pre-commit hooks"
        pre-commit install
        log_info "Pre-commit hooks installed"
    fi
}

# Print post-install instructions
print_post_install() {
    echo ""
    echo "================================================"
    log_info "Nexus-LLM installed successfully!"
    echo "================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Configure environment:"
    echo "     cp .env.example .env"
    echo "     # Edit .env with your settings"
    echo ""
    echo "  3. Start the server:"
    echo "     nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    echo "  4. Or use the Python SDK:"
    echo "     python -c \"from nexus_llm import NexusClient; print('Ready!')\""
    echo ""
    echo "Documentation: https://nexus-llm.readthedocs.io"
    echo "GitHub:        https://github.com/nexus-llm/nexus-llm"
    echo "Discord:       https://discord.gg/nexus-llm"
    echo ""
}

# Main
main() {
    check_python
    create_venv
    install_deps
    verify_installation
    install_precommit
    print_post_install
}

main
