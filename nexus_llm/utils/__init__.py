"""Nexus-LLM Utils Module.

Provides utility functions for logging, I/O, system info, networking,
cryptography, text processing, validation, decorators, retry logic,
threading, process management, timing, hashing, formatting,
sanitization, and profiling.
"""

from nexus_llm.utils.logger import setup_logger, get_logger
from nexus_llm.utils.helpers import MODEL_CATALOG, get_model_info, list_models_by_category
from nexus_llm.utils.io import (
    read_file,
    write_file,
    read_json,
    write_json,
    read_yaml,
    write_yaml,
    read_jsonl,
    write_jsonl,
    read_csv,
    write_csv,
    ensure_dir,
)
from nexus_llm.utils.system import get_system_info, get_gpu_info, get_cuda_version
from nexus_llm.utils.network import download_file, HttpClient
from nexus_llm.utils.crypto import compute_md5, compute_sha256, verify_file_integrity
from nexus_llm.utils.text import (
    count_tokens,
    count_words,
    split_sentences,
    truncate_text,
    clean_text,
)
from nexus_llm.utils.validation import (
    validate_type,
    validate_range,
    validate_path,
    validate_url,
)
from nexus_llm.utils.decorators import (
    timing,
    retry,
    cache_result,
    singleton,
    deprecated,
    validate_args,
)
from nexus_llm.utils.retry import RetryPolicy, with_retry
from nexus_llm.utils.threading import ThreadPool, BackgroundWorker, AsyncBridge
from nexus_llm.utils.process import run_subprocess, ProcessPool
from nexus_llm.utils.timer import Timer, ElapsedTime, ETACalculator
from nexus_llm.utils.hash import file_hash, content_hash, consistent_hash
from nexus_llm.utils.format import (
    bytes_to_human,
    seconds_to_duration,
    format_number,
    format_percentage,
)
from nexus_llm.utils.sanitize import (
    escape_html,
    sanitize_path,
    sanitize_filename,
)
from nexus_llm.utils.profiler import CPUProfiler, MemoryProfiler, FunctionProfiler

__all__ = [
    "setup_logger",
    "get_logger",
    "MODEL_CATALOG",
    "get_model_info",
    "list_models_by_category",
    "read_file",
    "write_file",
    "read_json",
    "write_json",
    "read_yaml",
    "write_yaml",
    "read_jsonl",
    "write_jsonl",
    "read_csv",
    "write_csv",
    "ensure_dir",
    "get_system_info",
    "get_gpu_info",
    "get_cuda_version",
    "download_file",
    "HttpClient",
    "compute_md5",
    "compute_sha256",
    "verify_file_integrity",
    "count_tokens",
    "count_words",
    "split_sentences",
    "truncate_text",
    "clean_text",
    "validate_type",
    "validate_range",
    "validate_path",
    "validate_url",
    "timing",
    "retry",
    "cache_result",
    "singleton",
    "deprecated",
    "validate_args",
    "RetryPolicy",
    "with_retry",
    "ThreadPool",
    "BackgroundWorker",
    "AsyncBridge",
    "run_subprocess",
    "ProcessPool",
    "Timer",
    "ElapsedTime",
    "ETACalculator",
    "file_hash",
    "content_hash",
    "consistent_hash",
    "bytes_to_human",
    "seconds_to_duration",
    "format_number",
    "format_percentage",
    "escape_html",
    "sanitize_path",
    "sanitize_filename",
    "CPUProfiler",
    "MemoryProfiler",
    "FunctionProfiler",
]
