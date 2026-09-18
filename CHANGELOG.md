# Changelog

All notable changes to the Nexus-LLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-09-11

### Fixed
- **Package import failures**: 10 modules raised on import (`agents.code_agent`,
  `agents.research_agent`, `agents.tool_agent`, `api.websocket`,
  `backend.pipeline`, `chains`, `evaluation.generation_eval`,
  `safety.moderation`, `safety.policies`, `terminal.autocomplete`) because they
  referenced symbols that did not exist. All 411 modules now import cleanly and
  CI enforces it with `scripts/dev/check_imports.py`.
- **Undefined names at import time** (silenced by the `F821` ignore in the ruff
  config): missing `Optional`/`Tuple`/`Callable`/`sys`/`datetime` imports in
  `tools/shell.py`, `tools/api_tool.py`, `rag/retriever.py`,
  `terminal/status.py`, `monitoring/*`, `training/distributed.py`; plus a broken
  `Event(...) if False else None` expression in `app.py`.
- `nexus_llm/cli_ext/output.py` was two copies of a module concatenated into one
  file and did not parse.
- **Streaming produced no output**: `TextIteratorStreamer`, `CallbackStreamer`
  and `AsyncStreamer` discarded the first `put()` and then sliced every payload
  by a prompt length that kept growing, so `model.generate(streamer=...)`
  yielded nothing. They now share an HF-compatible decoder and stream correctly
  for both per-step and full-sequence callers.
- **Chat markdown rendering dropped text after a heading**: `_split_blocks`
  never split on blank lines, so a heading block swallowed the rest of the
  response. Blocks now split on blank lines and headings get their own block.
- Panel subtitles longer than the body were truncated by Rich; the renderer
  sizes the surface to fit content, title and subtitle.
- `MetricsRegistry.to_dict()` called `Gauge.get_all()`, which did not exist, so
  `/metrics` raised `AttributeError`.
- `DataCollator(pad_to_multiple_of=...)` was ignored for list-based (JSONL)
  batches, and `OptimizerConfig(separate_decay_groups=False)` dropped the
  configured `weight_decay`, letting the optimizer default win silently.
- `list_checkpoints()` returns checkpoints ordered by training step.
- The chat converter emitted a corrupt marker (`<|assistant|)`); it is now
  `<|assistant|>`.
- `DatasetLoader` rejected plain `{"text": ...}` corpora; a `text` format was
  added so language-model datasets load without naming a format.
- Two rival `NexusLLMError` bases meant the CLI could not catch errors raised by
  library modules; `nexus_llm.core.exceptions` now shares the canonical root.
- `utils/crypto.py` used PEP 604 unions without the future import, breaking
  Python 3.9 (the declared floor).
- `nexus_llm/nexus/*` classes are re-exported under their `Nexus*` names;
  `safety`, `agents`, `evaluation` and `rag` export their public API.
- Added the missing `ActionExecutor` used by every agent, a `TokenizerWrapper`
  used by the inference pipeline, and `ConditionalChain`, which `chains`
  exported but never defined.

### Changed
- **Packaging**: `pyproject.toml` is the single source of truth (duplicate
  `setup.cfg` metadata removed), version is read dynamically from
  `nexus_llm/__version__.py`, and `VERSION`/`__init__.py` no longer drift.
  `license-files`, `project.urls`, and `train`/`quantization`/`gpu`/`rag`/
  `docs`/`all` extras are declared; PEP 639 license expression replaces the
  superseded license classifier.
- **Package data**: i18n catalogs, bundled presets, prompt templates and
  `py.typed` are installed with the wheel (previously lost by a non-editable
  install).
- Config no longer resolves `config/default_config.yaml` relative to
  `site-packages`, so `nexus-llm config` works after `pip install`.
- `requirements*.txt` match the package dependencies; GPU pins no longer
  combine `>=` with local `+cu121` versions that pip cannot satisfy.
- **CI workflows**: consolidated into `ci.yml` (lint, import check, test matrix
  3.9-3.12, wheel install smoke test, docker build, coverage), `docs.yml` and
  `release.yml`; duplicated `lint.yml`, `test.yml` and `python-publish.yml` were
  removed. Triggers now include `master` (the default branch) so jobs actually
  run, GPU tests are gated on a repository variable instead of queueing forever
  for a runner that may not exist, coverage combining no longer feeds XML to
  `coverage combine`, and the docs job no longer generates a broken heredoc.
- Dockerfile builds an image from the package (non-editable), runs as a
  non-root user, health-checks `/health`, and exposes a `CUDA_VERSION` build
  arg; compose splits GPU settings into `docker-compose.gpu.yml`.
- Makefile targets match CI (`make check`, `make install-dev`, lint/format,
  import and version checks) and point at the real script paths.

### Tests
- `tests/conftest.py` provides the `tmp_dir` fixture used by 57 test modules and
  a per-test event loop; `tox.ini`'s malformed `[tox] =` header (which broke
  pytest collection for the whole suite) is fixed.
- 30 test modules importing a non-existent `nexus.*` package were replaced by
  `tests/test_terminal_ansi.py`, `tests/test_terminal_ui.py`,
  `tests/test_backend_inference.py`, `tests/test_training_pipeline.py` and
  `tests/test_inference_e2e.py`, which exercise the real implementations
  (including generation and streaming against a tiny local GPT-2).

## [0.1.0] - 2024-01-15

### Added
- Initial release of Nexus-LLM framework
- CLI interface with commands: chat, serve, train, train-data, models, download, eval, benchmark, config
- Interactive chat mode with conversation history support
- FastAPI-based inference server with WebSocket support
- Model download and management system with 39+ supported models
- Fine-tuning support with LoRA/QLoRA via PEFT
- Training data preparation and processing pipeline
- Evaluation framework for benchmark tasks
- Benchmarking tools for inference performance measurement
- Configuration management from files, environment variables, and CLI
- Plugin system with extensible architecture
- Event bus for inter-component communication
- Component registry for models, plugins, and commands
- Custom exception hierarchy for structured error handling
- Type definitions with dataclasses and TypedDict
- Enums for model types, devices, precision, tasks, and chat roles
- Application context manager for resource lifecycle
- State management with StateManager
- Signal handling for graceful shutdown
- Docker support with multi-stage build and docker-compose
- GPU support with CUDA, xformers, and flash-attention
- Model quantization support (GPTQ, AWQ, bitsandbytes)
- Rich terminal output with progress bars and formatting
- Comprehensive logging system
- Makefile for common development tasks
- CI/CD configuration with tox
- Pre-commit hook configuration

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

## [Unreleased]

### Added
- Documentation site with Sphinx
- Model merging support
- Multi-model serving
- Streaming response support for server
- Batch inference API endpoint
- RLHF training support
- Distributed training with DeepSpeed
- OpenAI-compatible API server
- Model conversion tools (GGUF, ONNX)

[0.1.0]: https://github.com/brutal-45/Nexus-LLM/releases/tag/v0.1.0
[Unreleased]: https://github.com/brutal-45/Nexus-LLM/compare/v0.1.0...HEAD
