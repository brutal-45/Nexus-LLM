#!/usr/bin/env bash
# =============================================================================
# Nexus-LLM Cleanup Script 
# =============================================================================
# Cleans up build artifacts, caches, temporary files, and optionally models.
#
# Usage:
#   ./scripts/clean.sh                 # Clean caches and build artifacts
#   ./scripts/clean.sh --all           # Deep clean everything
#   ./scripts/clean.sh --models        # Also remove downloaded models
#   ./scripts/clean.sh --checkpoints   # Also remove training checkpoints
#   ./scripts/clean.sh --dry-run       # Show what would be deleted
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
CLEAN_ALL=0
CLEAN_MODELS=0
CLEAN_CHECKPOINTS=0
CLEAN_DATA=0
CLEAN_VENV=0
DRY_RUN=0
VERBOSE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)          CLEAN_ALL=1; shift ;;
        --models)       CLEAN_MODELS=1; shift ;;
        --checkpoints)  CLEAN_CHECKPOINTS=1; shift ;;
        --data)         CLEAN_DATA=1; shift ;;
        --venv)         CLEAN_VENV=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --verbose)      VERBOSE=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all           Deep clean everything (caches, models, checkpoints, data)"
            echo "  --models        Also remove downloaded model weights"
            echo "  --checkpoints   Also remove training checkpoints"
            echo "  --data          Also remove processed data and vector stores"
            echo "  --venv          Also remove virtual environment"
            echo "  --dry-run       Show what would be deleted without deleting"
            echo "  --verbose       Verbose output"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate directory size
dir_size() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "0"
        return
    fi
    if [[ "$OSTYPE" == "darwin"* ]]; then
        du -sm "$dir" 2>/dev/null | awk '{print $1}'
    else
        du -sm "$dir" 2>/dev/null | awk '{print $1}'
    fi
}

# Remove path (or show what would be removed)
remove_path() {
    local path="$1"
    local description="$2"
    
    if [ ! -e "$path" ]; then
        [ "$VERBOSE" -eq 1 ] && log_info "Not found: ${path}"
        return
    fi
    
    local size
    size=$(dir_size "$path")
    
    if [ "$DRY_RUN" -eq 1 ]; then
        log_info "[DRY RUN] Would remove: ${path} (~${size}MB) - ${description}"
    else
        log_info "Removing: ${path} (~${size}MB) - ${description}"
        rm -rf "$path"
    fi
}

# Clean Python caches
clean_python_caches() {
    log_step "Cleaning Python caches"
    
    # __pycache__ directories
    while IFS= read -r -d '' dir; do
        remove_path "$dir" "__pycache__"
    done < <(find "$PROJECT_ROOT" -type d -name "__pycache__" -print0 2>/dev/null || true)
    
    # .pyc files
    while IFS= read -r -d '' file; do
        remove_path "$file" ".pyc file"
    done < <(find "$PROJECT_ROOT" -type f -name "*.pyc" -print0 2>/dev/null || true)
    
    # .pyo files
    while IFS= read -r -d '' file; do
        remove_path "$file" ".pyo file"
    done < <(find "$PROJECT_ROOT" -type f -name "*.pyo" -print0 2>/dev/null || true)
    
    # .egg-info directories
    while IFS= read -r -d '' dir; do
        remove_path "$dir" "egg-info"
    done < <(find "$PROJECT_ROOT" -type d -name "*.egg-info" -print0 2>/dev/null || true)
}

# Clean build artifacts
clean_build_artifacts() {
    log_step "Cleaning build artifacts"
    
    remove_path "${PROJECT_ROOT}/build" "build directory"
    remove_path "${PROJECT_ROOT}/dist" "dist directory"
    remove_path "${PROJECT_ROOT}/*.egg" "egg files"
}

# Clean test artifacts
clean_test_artifacts() {
    log_step "Cleaning test artifacts"
    
    remove_path "${PROJECT_ROOT}/.pytest_cache" "pytest cache"
    remove_path "${PROJECT_ROOT}/.mypy_cache" "mypy cache"
    remove_path "${PROJECT_ROOT}/.ruff_cache" "ruff cache"
    remove_path "${PROJECT_ROOT}/.pytype" "pytype cache"
    remove_path "${PROJECT_ROOT}/htmlcov" "coverage HTML"
    remove_path "${PROJECT_ROOT}/coverage.xml" "coverage XML"
    remove_path "${PROJECT_ROOT}/.coverage" "coverage data"
    remove_path "${PROJECT_ROOT}/test-results" "test results"
}

# Clean tool caches
clean_tool_caches() {
    log_step "Cleaning tool caches"
    
    remove_path "${PROJECT_ROOT}/.cache" "project cache"
    remove_path "${PROJECT_ROOT}/nexus_cache" "nexus cache"
    remove_path "${PROJECT_ROOT}/.faiss" "FAISS index cache"
    remove_path "${PROJECT_ROOT}/.huggingface" "HF cache"
    remove_path "${PROJECT_ROOT}/.transformers_cache" "transformers cache"
}

# Clean logs
clean_logs() {
    log_step "Cleaning logs"
    
    remove_path "${PROJECT_ROOT}/logs" "log files"
    
    while IFS= read -r -d '' file; do
        remove_path "$file" "log file"
    done < <(find "$PROJECT_ROOT" -maxdepth 1 -type f -name "*.log" -print0 2>/dev/null || true)
}

# Clean models
clean_models() {
    if [ "$CLEAN_MODELS" -eq 0 ] && [ "$CLEAN_ALL" -eq 0 ]; then
        return
    fi
    
    log_step "Cleaning model weights"
    log_warn "This will delete downloaded model weights!"
    
    if [ "$DRY_RUN" -eq 0 ]; then
        read -rp "Delete all models? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping model cleanup"
            return
        fi
    fi
    
    remove_path "${PROJECT_ROOT}/models" "model weights"
}

# Clean checkpoints
clean_checkpoints() {
    if [ "$CLEAN_CHECKPOINTS" -eq 0 ] && [ "$CLEAN_ALL" -eq 0 ]; then
        return
    fi
    
    log_step "Cleaning training checkpoints"
    
    if [ "$DRY_RUN" -eq 0 ]; then
        read -rp "Delete all checkpoints? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping checkpoint cleanup"
            return
        fi
    fi
    
    remove_path "${PROJECT_ROOT}/checkpoints" "training checkpoints"
    remove_path "${PROJECT_ROOT}/runs" "tensorboard runs"
    remove_path "${PROJECT_ROOT}/lightning_logs" "lightning logs"
    remove_path "${PROJECT_ROOT}/mlruns" "MLflow runs"
    remove_path "${PROJECT_ROOT}/wandb" "W&B runs"
}

# Clean data
clean_data() {
    if [ "$CLEAN_DATA" -eq 0 ] && [ "$CLEAN_ALL" -eq 0 ]; then
        return
    fi
    
    log_step "Cleaning processed data"
    
    remove_path "${PROJECT_ROOT}/data/processed" "processed data"
    remove_path "${PROJECT_ROOT}/data/vector_store" "vector store data"
}

# Clean virtual environment
clean_venv() {
    if [ "$CLEAN_VENV" -eq 0 ] && [ "$CLEAN_ALL" -eq 0 ]; then
        return
    fi
    
    log_step "Cleaning virtual environment"
    remove_path "${PROJECT_ROOT}/.venv" "virtual environment"
}

# Print summary
print_summary() {
    echo ""
    echo "============================================"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  Clean Summary (DRY RUN - nothing deleted)"
    else
        echo "  Clean Summary"
    fi
    echo "============================================"
    echo ""
    
    local total_freed=0
    for dir in build dist .pytest_cache .mypy_cache .ruff_cache htmlcov logs; do
        if [ -d "${PROJECT_ROOT}/${dir}" ]; then
            size=$(dir_size "${PROJECT_ROOT}/${dir}")
            total_freed=$((total_freed + size))
        fi
    done
    
    if [ "$DRY_RUN" -eq 0 ]; then
        log_info "Estimated space freed: ~${total_freed}MB"
    else
        log_info "Estimated space that would be freed: ~${total_freed}MB"
    fi
    echo ""
}

# Main
main() {
    echo ""
    echo "============================================"
    echo "  Nexus-LLM Cleanup"
    echo "============================================"
    echo ""
    
    if [ "$DRY_RUN" -eq 1 ]; then
        log_info "DRY RUN mode - nothing will be deleted"
        echo ""
    fi
    
    clean_python_caches
    clean_build_artifacts
    clean_test_artifacts
    clean_tool_caches
    clean_logs
    clean_models
    clean_checkpoints
    clean_data
    clean_venv
    
    print_summary
}

main
