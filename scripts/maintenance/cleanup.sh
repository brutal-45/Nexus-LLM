#!/bin/bash
# Cleanup Old Files - Nexus-LLM
# ==============================
# Removes old temporary files, logs, caches, and stale data.
#
# Usage:
#   ./cleanup.sh [OPTIONS]
#
# Options:
#   -d, --days N          Remove files older than N days (default: 30)
#   -l, --logs            Clean up log files
#   -c, --cache           Clean up cache directories
#   -t, --temp            Clean up temporary files
#   -m, --models-cache    Clean up cached model files (use with caution)
#   -b, --benchmark       Clean up old benchmark results
#   -a, --all             Clean up everything
#   -n, --dry-run         Show what would be deleted without deleting
#   -v, --verbose         Verbose output
#   -h, --help            Show this help message

set -euo pipefail

# Default values
DAYS=30
CLEAN_LOGS=false
CLEAN_CACHE=false
CLEAN_TEMP=false
CLEAN_MODELS=false
CLEAN_BENCHMARK=false
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--days)
            DAYS="$2"
            shift 2
            ;;
        -l|--logs)
            CLEAN_LOGS=true
            shift
            ;;
        -c|--cache)
            CLEAN_CACHE=true
            shift
            ;;
        -t|--temp)
            CLEAN_TEMP=true
            shift
            ;;
        -m|--models-cache)
            CLEAN_MODELS=true
            shift
            ;;
        -b|--benchmark)
            CLEAN_BENCHMARK=true
            shift
            ;;
        -a|--all)
            CLEAN_LOGS=true
            CLEAN_CACHE=true
            CLEAN_TEMP=true
            CLEAN_BENCHMARK=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
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

# If no specific option is set, clean logs and temp by default
if [[ "${CLEAN_LOGS}" == false && "${CLEAN_CACHE}" == false && \
      "${CLEAN_TEMP}" == false && "${CLEAN_MODELS}" == false && \
      "${CLEAN_BENCHMARK}" == false ]]; then
    CLEAN_LOGS=true
    CLEAN_TEMP=true
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOTAL_SIZE=0
TOTAL_FILES=0

log() {
    if [[ "${VERBOSE}" == true ]]; then
        echo "  $1"
    fi
}

calculate_size() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        du -sb "${path}" 2>/dev/null | cut -f1
    else
        echo 0
    fi
}

clean_directory() {
    local dir="$1"
    local pattern="$2"
    local description="$3"

    if [[ ! -d "${dir}" ]]; then
        log "Directory does not exist: ${dir}"
        return
    fi

    echo "Cleaning ${description} (older than ${DAYS} days)..."

    local count=0
    local size=0

    while IFS= read -r -d '' file; do
        file_size=$(stat -c%s "${file}" 2>/dev/null || echo 0)
        size=$((size + file_size))
        count=$((count + 1))

        if [[ "${DRY_RUN}" == true ]]; then
            log "[DRY RUN] Would remove: ${file}"
        else
            rm -f "${file}"
            log "Removed: ${file}"
        fi
    done < <(find "${dir}" -name "${pattern}" -mtime +"${DAYS}" -print0 2>/dev/null)

    TOTAL_SIZE=$((TOTAL_SIZE + size))
    TOTAL_FILES=$((TOTAL_FILES + count))

    local size_mb=$((size / 1024 / 1024))
    if [[ "${DRY_RUN}" == true ]]; then
        echo "  Would remove ${count} files (${size_mb} MB)"
    else
        echo "  Removed ${count} files (${size_mb} MB)"
    fi
}

echo "=============================================="
echo "Nexus-LLM Cleanup"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Max age:      ${DAYS} days"
echo "Dry run:      ${DRY_RUN}"
echo "=============================================="
echo ""

# Clean log files
if [[ "${CLEAN_LOGS}" == true ]]; then
    clean_directory "${PROJECT_ROOT}/logs" "*.log" "log files"
    clean_directory "${PROJECT_ROOT}/logs" "*.log.*" "rotated log files"
fi

# Clean cache directories
if [[ "${CLEAN_CACHE}" == true ]]; then
    clean_directory "${PROJECT_ROOT}/.cache" "*" "cache files"
    clean_directory "${PROJECT_ROOT}/data/.cache" "*" "data cache"
    # Clean __pycache__ directories
    echo "Cleaning Python cache directories..."
    pycache_count=0
    while IFS= read -r -d '' dir; do
        if [[ "${DRY_RUN}" == true ]]; then
            log "[DRY RUN] Would remove: ${dir}"
        else
            rm -rf "${dir}"
            log "Removed: ${dir}"
        fi
        pycache_count=$((pycache_count + 1))
    done < <(find "${PROJECT_ROOT}" -type d -name "__pycache__" -mtime +"${DAYS}" -print0 2>/dev/null)
    echo "  Processed ${pycache_count} __pycache__ directories"
fi

# Clean temporary files
if [[ "${CLEAN_TEMP}" == true ]]; then
    clean_directory "${PROJECT_ROOT}/tmp" "*" "temporary files"
    clean_directory "${PROJECT_ROOT}/temp" "*" "temporary files"
    # Clean .tmp files throughout the project
    clean_directory "${PROJECT_ROOT}" "*.tmp" "temp files"
    clean_directory "${PROJECT_ROOT}" "*.swp" "vim swap files"
fi

# Clean benchmark results
if [[ "${CLEAN_BENCHMARK}" == true ]]; then
    clean_directory "${PROJECT_ROOT}/benchmark_results" "*.json" "benchmark results"
    clean_directory "${PROJECT_ROOT}/benchmark_results" "*.csv" "benchmark CSVs"
fi

# Clean model cache (use with caution!)
if [[ "${CLEAN_MODELS}" == true ]]; then
    echo "WARNING: Cleaning model cache! This will require re-downloading models."
    if [[ "${DRY_RUN}" == false ]]; then
        read -p "Are you sure? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping model cache cleanup."
        else
            clean_directory "${PROJECT_ROOT}/models/.cache" "*" "cached model files"
        fi
    else
        clean_directory "${PROJECT_ROOT}/models/.cache" "*" "cached model files"
    fi
fi

# Summary
echo ""
echo "=============================================="
echo "Cleanup Summary"
echo "=============================================="
total_size_mb=$((TOTAL_SIZE / 1024 / 1024))
if [[ "${DRY_RUN}" == true ]]; then
    echo "Would remove: ${TOTAL_FILES} files (${total_size_mb} MB)"
    echo ""
    echo "Run without --dry-run to actually delete these files."
else
    echo "Removed: ${TOTAL_FILES} files (${total_size_mb} MB)"
fi
echo "=============================================="
