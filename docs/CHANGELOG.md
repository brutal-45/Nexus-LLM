# Changelog

All notable changes to Nexus-LLM are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [2.1.0] - 2024-12-15

### Added
- Mamba-based architecture support alongside transformer models
- GPTQ and AWQ quantization methods for 4-bit and 8-bit inference
- Streaming support for RAG pipelines
- Cross-encoder reranking for improved retrieval quality
- DPO alignment training pipeline
- Multimodal pipeline for image and audio inputs
- i18n module with 10 supported locales
- Plugin system with pre/post inference hooks
- Chain pipelines with conditional branching and parallel execution
- WebSocket API for real-time streaming
- Prometheus metrics exporter
- Built-in safety filter with toxicity detection and PII redaction
- Prompt injection detection

### Changed
- Improved continuous batching algorithm (up to 3x throughput increase)
- Updated FAISS integration to support GPU-accelerated indexing
- Refactored conversation management for better memory efficiency
- Enhanced error messages with actionable suggestions
- Upgraded minimum Python version to 3.9

### Fixed
- Memory leak in long-running inference sessions
- Race condition in concurrent model loading
- Incorrect token counting for multilingual text
- Streaming callback not invoked on error
- KV-cache corruption with very long sequences (>32K tokens)

## [2.0.0] - 2024-09-01

### Added
- Complete rewrite of the inference engine for modularity
- Model registry with automatic hardware detection
- Agent framework with tool registration and reasoning loop
- RAG pipeline with document ingestion, chunking, and vector indexing
- REST API server with authentication and rate limiting
- Conversation export in JSON, Markdown, HTML, and fine-tuning JSONL
- Benchmark suite with standard benchmarks (MMLU, HumanEval, GSM8K)
- Custom evaluation metrics framework
- Monitoring dashboard with real-time metrics

### Changed
- **BREAKING**: New `InferenceEngine` API replaces `Model` class
- **BREAKING**: Configuration files now use YAML instead of JSON
- **BREAKING**: Plugin interface updated with typed hook decorators

### Removed
- Legacy `Model` class (use `InferenceEngine`)
- `SimpleServer` class (use `NexusServer`)
- Python 3.7 and 3.8 support

## [1.5.0] - 2024-06-15

### Added
- LoRA fine-tuning support
- Flash Attention 2 integration
- Conversation persistence and loading
- Custom prompt template library
- Batch inference API

### Changed
- 2x faster tokenization with Rust-based tokenizer
- Reduced GPU memory usage by 30% with KV-cache optimization

### Fixed
- Timeout handling in streaming mode
- Incorrect batch size handling on multi-GPU setups

## [1.4.0] - 2024-04-01

### Added
- Multi-GPU inference with tensor parallelism
- Quantization support (INT8 via bitsandbytes)
- Health check endpoint for the API server
- Request queuing with priority levels

### Fixed
- Segfault with certain model architectures on Ampere GPUs
- Incorrect gradient computation in mixed-precision training

## [1.3.0] - 2024-02-01

### Added
- Streaming token generation
- Conversation context management
- Basic RAG pipeline (document retrieval + generation)
- Docker support with pre-built images

### Changed
- Improved API server throughput by 50%

## [1.2.0] - 2023-12-01

### Added
- REST API server
- Authentication with API keys
- Rate limiting middleware

## [1.1.0] - 2023-10-01

### Added
- Chat conversation support
- System prompts
- Temperature and top-p sampling controls

## [1.0.0] - 2023-08-01

### Added
- Initial release
- Basic text generation with HuggingFace models
- Single-turn inference API
- GPU and CPU support
