# Changelog

All notable changes to the Nexus-LLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
