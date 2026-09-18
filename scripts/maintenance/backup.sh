#!/bin/bash
# Backup Project - Nexus-LLM
# ============================
# Creates a backup of the Nexus-LLM project, including configuration,
# data, and model adapters (but not base model weights).
#
# Usage:
#   ./backup.sh [OPTIONS]
#
# Options:
#   -o, --output DIR        Output directory for backups (default: ./backups)
#   -e, --exclude PATTERNS  Additional exclude patterns (comma-separated)
#   -c, --config-only       Only backup configuration files
#   -f, --full              Full backup including model weights
#   -z, --compress          Compress the backup with gzip
#   -k, --keep N            Keep only the N most recent backups (default: 5)
#   -v, --verbose           Verbose output
#   -h, --help              Show this help message

set -euo pipefail

# Default values
OUTPUT_DIR="./backups"
EXCLUDE=""
CONFIG_ONLY=false
FULL=false
COMPRESS=true
KEEP=5
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -e|--exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        -c|--config-only)
            CONFIG_ONLY=true
            shift
            ;;
        -f|--full)
            FULL=true
            shift
            ;;
        -z|--compress)
            COMPRESS=true
            shift
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        -k|--keep)
            KEEP="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -15 "$0" | tail -13
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="nexus-llm-backup-${TIMESTAMP}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Nexus-LLM Backup"
echo "=============================================="
echo "Project root:   ${PROJECT_ROOT}"
echo "Backup name:    ${BACKUP_NAME}"
echo "Output:         ${OUTPUT_DIR}"
echo "Config only:    ${CONFIG_ONLY}"
echo "Full backup:    ${FULL}"
echo "Compress:       ${COMPRESS}"
echo "Keep last:      ${KEEP}"
echo "=============================================="
echo ""

# Build the tar command
TAR_ARGS=(
    --create
    --file "${OUTPUT_DIR}/${BACKUP_NAME}.tar"
    --directory "${PROJECT_ROOT}"
    --verbose="${VERBOSE}"
)

# Exclude patterns (always exclude these)
DEFAULT_EXCLUDES=(
    "__pycache__"
    "*.pyc"
    "*.pyo"
    ".git"
    ".gitignore"
    ".env"
    "node_modules"
    ".venv"
    "venv"
    "*.egg-info"
    "dist"
    "build"
    ".mypy_cache"
    ".pytest_cache"
    ".ruff_cache"
)

# Exclude model weights unless --full is specified
if [[ "${FULL}" == false ]]; then
    DEFAULT_EXCLUDES+=(
        "models/*/pytorch_model*.bin"
        "models/*/model*.safetensors"
        "models/*/tf_model*"
        "models/*/flax_model*"
    )
fi

# Exclude data unless --full is specified
if [[ "${CONFIG_ONLY}" == true ]]; then
    DEFAULT_EXCLUDES+=(
        "data"
        "benchmark_results"
        "logs"
        "tmp"
        "temp"
        "exports"
        "checkpoints"
    )
fi

# Add exclude arguments
for pattern in "${DEFAULT_EXCLUDES[@]}"; do
    TAR_ARGS+=(--exclude="${pattern}")
done

# Add user-provided excludes
if [[ -n "${EXCLUDE}" ]]; then
    IFS=',' read -ra USER_EXCLUDES <<< "${EXCLUDE}"
    for pattern in "${USER_EXCLUDES[@]}"; do
        TAR_ARGS+=(--exclude="${pattern}")
    done
fi

# Items to include
INCLUDE_ITEMS=(
    "config"
    "docs"
    "examples"
    "notebooks"
    "scripts"
    "nexus_llm"
    "setup.py"
    "setup.cfg"
    "pyproject.toml"
    "requirements.txt"
    "README.md"
    "LICENSE"
)

if [[ "${CONFIG_ONLY}" == false ]]; then
    INCLUDE_ITEMS+=(
        "data"
        "checkpoints"
        "logs"
    )
fi

if [[ "${FULL}" == true ]]; then
    INCLUDE_ITEMS+=("models")
fi

# Add include arguments
for item in "${INCLUDE_ITEMS[@]}"; do
    if [[ -e "${PROJECT_ROOT}/${item}" ]]; then
        TAR_ARGS+=("${item}")
    fi
done

# Create the backup
echo "Creating backup..."
echo ""

tar "${TAR_ARGS[@]}" 2>/dev/null || {
    # Fallback: just backup what exists
    echo "Some items not found, backing up available files..."
    EXISTING_ITEMS=()
    for item in "${INCLUDE_ITEMS[@]}"; do
        if [[ -e "${PROJECT_ROOT}/${item}" ]]; then
            EXISTING_ITEMS+=("${item}")
        fi
    done

    tar --create \
        --file "${OUTPUT_DIR}/${BACKUP_NAME}.tar" \
        --directory "${PROJECT_ROOT}" \
        "${EXISTING_ITEMS[@]}" \
        2>/dev/null || true
}

# Compress if requested
if [[ "${COMPRESS}" == true ]]; then
    echo "Compressing backup..."
    gzip -f "${OUTPUT_DIR}/${BACKUP_NAME}.tar"
    BACKUP_FILE="${OUTPUT_DIR}/${BACKUP_NAME}.tar.gz"
else
    BACKUP_FILE="${OUTPUT_DIR}/${BACKUP_NAME}.tar"
fi

# Calculate backup size
BACKUP_SIZE=$(du -sh "${BACKUP_FILE}" | cut -f1)

echo ""
echo "=============================================="
echo "Backup Complete!"
echo "=============================================="
echo "File:     ${BACKUP_FILE}"
echo "Size:     ${BACKUP_SIZE}"
echo ""

# Generate checksum
echo "Generating checksum..."
sha256sum "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"
echo "Checksum: $(cat "${BACKUP_FILE}.sha256")"
echo ""

# Clean up old backups
echo "Cleaning up old backups (keeping last ${KEEP})..."
ls -t "${OUTPUT_DIR}"/nexus-llm-backup-*.tar* 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old_backup; do
    echo "  Removing: ${old_backup}"
    rm -f "${old_backup}" "${old_backup}.sha256"
done

echo ""
echo "Available backups:"
ls -lh "${OUTPUT_DIR}"/nexus-llm-backup-*.tar* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  None"
echo "=============================================="
