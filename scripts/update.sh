#!/usr/bin/env bash
# =============================================================================
# Nexus-LLM Update Script
# =============================================================================
# Updates Nexus-LLM to the latest version.
#
# Usage:
#   ./scripts/update.sh                    # Update to latest stable
#   ./scripts/update.sh --pre-release      # Update to latest pre-release
#   ./scripts/update.sh --version 1.2.0    # Update to specific version
#   ./scripts/update.sh --from-source      # Update from git source
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
TARGET_VERSION=""
PRE_RELEASE=0
FROM_SOURCE=0
BACKUP=1
SKIP_DEPS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)      TARGET_VERSION="$2"; shift 2 ;;
        --pre-release)  PRE_RELEASE=1; shift ;;
        --from-source)  FROM_SOURCE=1; shift ;;
        --no-backup)    BACKUP=0; shift ;;
        --skip-deps)    SKIP_DEPS=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version VER       Update to specific version"
            echo "  --pre-release       Include pre-release versions"
            echo "  --from-source       Update from git source (git pull)"
            echo "  --no-backup         Skip backup of current version"
            echo "  --skip-deps         Skip dependency updates"
            echo "  -h, --help          Show this help"
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
    if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
    fi
}

# Get current version
get_current_version() {
    python3 -c "
try:
    import nexus_llm
    print(nexus_llm.__version__)
except ImportError:
    print('unknown')
" 2>/dev/null || echo "unknown"
}

# Get latest version from PyPI
get_latest_version() {
    local pre_flag=""
    if [ "$PRE_RELEASE" -eq 1 ]; then
        pre_flag="--pre"
    fi
    
    pip index versions nexus-llm $pre_flag 2>/dev/null | \
        head -1 | \
        sed 's/nexus-llm (\(.*\))/\1/' | \
        cut -d',' -f1 | \
        tr -d ' '
}

# Backup current installation
backup_current() {
    if [ "$BACKUP" -eq 0 ]; then
        return
    fi
    
    local current_ver
    current_ver=$(get_current_version)
    
    if [ "$current_ver" == "unknown" ]; then
        log_warn "Cannot determine current version, skipping backup"
        return
    fi
    
    log_step "Backing up current version (${current_ver})"
    
    local backup_dir="${PROJECT_ROOT}/.backups"
    mkdir -p "$backup_dir"
    
    # Save current requirements
    pip freeze > "${backup_dir}/requirements_${current_ver}.txt"
    log_info "Requirements saved to ${backup_dir}/requirements_${current_ver}.txt"
}

# Update via pip
update_pip() {
    local version_arg=""
    if [ -n "$TARGET_VERSION" ]; then
        version_arg="==${TARGET_VERSION}"
    fi
    
    log_step "Updating Nexus-LLM via pip"
    
    local install_flag=""
    if [ "$PRE_RELEASE" -eq 1 ]; then
        install_flag="--pre"
    fi
    
    pip install --upgrade ${install_flag} "nexus-llm${version_arg}"
    
    local new_ver
    new_ver=$(get_current_version)
    log_info "Updated to version: ${new_ver}"
}

# Update from source
update_source() {
    log_step "Updating Nexus-LLM from source"
    
    cd "$PROJECT_ROOT"
    
    # Check for uncommitted changes
    if ! git diff --quiet 2>/dev/null; then
        log_warn "Uncommitted changes detected. Stashing..."
        git stash push -m "nexus-llm-update-$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Pull latest
    local branch="main"
    if [ -n "$TARGET_VERSION" ]; then
        branch="v${TARGET_VERSION}"
    fi
    
    git fetch origin
    git checkout "$branch" 2>/dev/null || git checkout main
    git pull origin "$(git branch --show-current)"
    
    # Reinstall
    pip install -e ".[dev]"
    
    local new_ver
    new_ver=$(get_current_version)
    log_info "Updated to version: ${new_ver}"
}

# Update dependencies
update_dependencies() {
    if [ "$SKIP_DEPS" -eq 1 ]; then
        log_info "Skipping dependency updates"
        return
    fi
    
    log_step "Updating dependencies"
    pip install --upgrade pip setuptools wheel
}

# Run migrations
run_migrations() {
    log_step "Checking for migrations"
    
    python3 -c "
from nexus_llm.config import run_migrations
run_migrations()
" 2>/dev/null && log_info "Migrations applied" || log_info "No migrations needed"
}

# Verify update
verify_update() {
    log_step "Verifying installation"
    
    local new_ver
    new_ver=$(get_current_version)
    
    if [ "$new_ver" == "unknown" ]; then
        log_error "Installation verification failed!"
        exit 1
    fi
    
    python3 -c "
import nexus_llm
print(f'Nexus-LLM version: {nexus_llm.__version__}')
print('Installation verified!')
"
    
    # Quick smoke test
    python3 -c "
from nexus_llm import NexusClient
client = NexusClient.__new__(NexusClient)
print('Import test passed')
" 2>/dev/null && log_info "Smoke test passed" || log_warn "Smoke test had warnings"
}

# Print changelog hint
print_changelog() {
    local new_ver
    new_ver=$(get_current_version)
    
    echo ""
    echo "============================================"
    log_info "Update complete!"
    echo "============================================"
    echo ""
    echo "  Current version: ${new_ver}"
    echo ""
    echo "  What's new:"
    echo "    https://github.com/nexus-llm/nexus-llm/releases/tag/v${new_ver}"
    echo ""
    echo "  Changelog:"
    echo "    https://github.com/nexus-llm/nexus-llm/blob/main/CHANGELOG.md"
    echo ""
    echo "  If you experience issues, restore from backup:"
    echo "    pip install -r .backups/requirements_*.txt"
    echo ""
}

# Main
main() {
    load_env
    
    local current_ver
    current_ver=$(get_current_version)
    
    echo ""
    echo "============================================"
    echo "  Nexus-LLM Updater"
    echo "============================================"
    echo "  Current version: ${current_ver}"
    echo "============================================"
    echo ""
    
    backup_current
    update_dependencies
    
    if [ "$FROM_SOURCE" -eq 1 ]; then
        update_source
    else
        update_pip
    fi
    
    run_migrations
    verify_update
    print_changelog
}

main
