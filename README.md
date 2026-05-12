<p align="center">
<h1 align="center">⬡ Nexus LLM</h1>
<p align="center">
  <strong>A Production-Grade 100B+ Parameter Large Language Model — Built From Scratch</strong>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/LoC-150K+-orange.svg" alt="Lines of Code">
  <img src="https://img.shields.io/badge/Files-130+-blue.svg" alt="Files">
</p>
</p>

---

## Overview

**Nexus** is a complete, open-source large language model codebase built entirely from scratch with PyTorch. It implements every component needed to train, fine-tune, serve, and deploy production-grade LLMs — from the mathematical foundations to the inference server — without relying on any external AI frameworks or model zoos.

The project is designed as a **research platform** and **production toolkit** for teams working on large-scale language models. Every module is self-contained, well-documented, and implements state-of-the-art algorithms from the ground up.

### Key Design Principles

- **From Scratch**: Every algorithm is implemented from first principles — no wrappers around external model libraries
- **Production-Ready**: Distributed training, quantization, serving, monitoring — everything needed for deployment
- **Extensible**: Modular architecture makes it easy to swap components, add new features, or experiment
- **Well-Documented**: Comprehensive docstrings, type hints, and inline explanations throughout
- **Zero External AI Dependencies**: No external model APIs — runs entirely on your hardware

---

## Features

### 🧠 Model Architecture
- **Decoder-Only Transformer** with configurable dimensions (d_model, heads, layers, FFN)
- **Attention Variants**: Multi-Head (MHA), Grouped-Query (GQA), Multi-Query (MQA), Multi-Latent (MLA), Differential
- **Flash Attention v2/v3**: Memory-efficient exact attention with CUDA kernels
- **Sparse Attention**: Longformer, BigBird, block sparse patterns
- **FFN Variants**: SwiGLU, GeGLU, ReGLU, GatedGELU, standard
- **Mixture of Experts (MoE)**: Top-K routing, load balancing, expert parallelism, 256+ experts
- **State Space Models**: S4, Mamba, RWKV — transformer alternatives
- **Positional Encodings**: RoPE with NTK/YaRN/LongRoPE/Dynamic scaling, ALiBi, learned
- **Embeddings**: Token, position, rotary, with scaling and interpolation
- **Normalizations**: RMSNorm, DeepNorm, QK-Norm, LayerNorm variants
- **Activations**: 15+ activation functions (GELU, SwiGLU, Mish, SiLU, StarReLU, etc.)
- **Residual Connections**: Pre-norm, post-norm, parallel, scaled, gated, highway
- **Architectural Innovations**: Parallel attention+FFN, Mixture of Depths, early exit, Universal Transformer

### ⚡ Training Infrastructure
- **9 Optimizers**: SGD, Adam, AdamW, LAMB, Adafactor, LION, Sophia, Shampoo, 8-bit Adam
- **11 LR Schedulers**: Cosine, Linear, Polynomial, Warmup, OneCycle, Cyclic, CosineAnnealing, Step, Exponential, Plateau, InverseSqrt
- **3D Parallelism**: Tensor Parallel, Pipeline Parallel, Data Parallel with ZeRO (Stages 1/2/3)
- **FSDP**: Fully Sharded Data Parallel with CPU offloading
- **BF16/FP16 Mixed Precision**: Dynamic loss scaling, gradient checkpointing
- **Gradient Accumulation**: With bucketed gradient reduction
- **Curriculum Learning**: Linear, step, competence-based, self-paced
- **Self-Play Training**: Adversarial generation, red-teaming, data augmentation
- **Constitutional AI**: Principle-based critique and revision
- **Hyperparameter Optimization**: Grid, Random, Bayesian (GP), Hyperband, ASHA, PBT, CMA-ES, TPE
- **Progressive Training**: Gradual unfreezing, batch size warmup, sequence length ramp
- **Convergence Analysis**: Loss landscape visualization, saddle point detection, critical batch size

### 🔮 Inference & Serving
- **Paged Attention**: Virtual memory management for KV cache
- **Speculative Decoding**: Draft models, self-speculative, Medusa heads, EAGLE
- **Multi-Token Prediction**: Predict N future tokens with geometric decay loss
- **Quantization**: GPTQ, AWQ, NF4 (bitsandbytes), FP8 E4M3/E5M2, SmoothQuant, mixed-precision
- **KV Cache**: Quantized (FP8/INT8/INT4), sliding window, cross-layer sharing, prefix caching
- **Continuous Batching**: Dynamic batch scheduling for serving
- **Streaming Inference**: Server-sent events for real-time token streaming
- **Distributed Serving**: Multi-worker load balancing, caching, rate limiting

### 🎯 Alignment
- **SFT**: Supervised Fine-Tuning with instruction tuning datasets
- **DPO**: Direct Preference Optimization with variants (IPO, KTO, SimPO, ORPO)
- **PPO**: Proximal Policy Optimization with GAE, adaptive KL, reward model integration
- **Reward Modeling**: Binary classification + multi-label reward models

### 🖼️ Multimodal
- **Vision Encoders**: ViT, SigLIP, ConvNeXt, EfficientViT
- **Audio Encoders**: Whisper-style, Mel Spectrogram (from raw waveforms), HuBERT
- **Video Encoders**: TimeSformer, VideoSwin, ViViT with tubelet embeddings
- **Cross-Modal Fusion**: Cross-attention, co-attention, gated, compact bilinear, adaptive
- **Projectors**: MLP (LLaVA), Q-Former (BLIP-2), Resampler (Perceiver), C-Abstractor
- **Training**: Contrastive loss, alignment loss, modality-balanced sampling

### 🤖 Agents & Reasoning
- **Chain-of-Thought**: Zero-shot, few-shot, auto-CoT, structured
- **Tree-of-Thought**: MCTS, beam search, BFS/DFS exploration
- **Planning**: Goal decomposition, dependency analysis, hierarchical planning
- **Verification**: Self, cross, backward, execution-based, consensus
- **Self-Consistency**: Multi-sample generation with majority voting
- **Tool Use**: Function calling with safety validation, sandboxing
- **Multi-Agent**: Collaborative, debate, hierarchical, ensemble architectures
- **Memory**: Long-term (vector-indexed), episodic, semantic, working memory

### 🌐 Distributed Training
- **Tensor Parallelism**: Column/row parallel layers with communication overlap
- **Pipeline Parallelism**: 1F1B and interleaved schedules, automatic load balancing
- **Data Parallelism**: DDP, ZeRO-1/2/3, FSDP with CPU offloading
- **Collective Operations**: All-reduce, all-gather, reduce-scatter with compression
- **Fault Tolerance**: Health checking, distributed checkpointing, auto-recovery
- **Elastic Training**: Dynamic scaling, node management, automatic rebalancing
- **RPC Framework**: Remote module execution, parameter server

### 🗜️ Optimization
- **Quantization**: GPTQ (2nd order), AWQ (activation-aware), NF4, FP8, SmoothQuant
- **Pruning**: Magnitude, structured (channel/head/FFN), SparseGPT, Wanda, LoRA pruning
- **Knowledge Distillation**: KL, feature, attention, multi-teacher, progressive
- **Neural Architecture Search**: Evolutionary, DARTS, random search, Bayesian, one-shot
- **Model Compression**: Weight sharing, low-rank factorization, layer fusion
- **Mixed Precision**: BF16/FP16 training with dynamic loss scaling
- **Gradient Checkpointing**: Block, selective, sequential recomputation policies
- **Memory Optimization**: CPU offloading, activation offloading, memory pooling
- **Compilation**: torch.compile (inductor/triton), CUDA graphs, Triton kernels

### 🧪 Knowledge & Memory
- **Retrieval**: Dense, sparse (BM25), hybrid, ColBERT
- **Reranking**: Cross-encoder, MMR diversity, ListNet, ensemble
- **Vector Stores**: Simple, HNSW, IVF, Product Quantization (all from scratch)
- **RAG Pipeline**: End-to-end retrieval-augmented generation with citations
- **Knowledge Graphs**: Entity/relation storage, TransE/TransH/TransR/ComplEx/RotatE embeddings
- **Document Processing**: Loaders (text/PDF/HTML/JSON/CSV), chunking strategies
- **Long-Term Memory**: Vector-indexed persistent storage with decay
- **Episodic Memory**: Experience recording and similarity-based recall
- **Semantic Memory**: Fact storage with verification and contradiction detection

### 🛡️ Safety
- **Content Safety Classifier**: Binary and multi-label classification
- **Guardrails**: Input and output filtering with configurable rules
- **Prompt Injection Detection**: Pattern and model-based detection
- **Jailbreak Detection**: Multiple defense strategies
- **PII Detection**: Personal identifiable information detection
- **Toxicity Detection**: Multi-category toxicity classification
- **Constitutional AI**: Principle-based alignment
- **Red Team Framework**: Automated adversarial testing
- **Bias Evaluation**: Fairness metrics across demographic groups
- **Differential Privacy**: Privacy-preserving training
- **Watermarking**: Model output watermarking

### 📊 Math Foundations
- **Autodiff**: Computational graph, forward/backward pass, automatic differentiation
- **Linear Algebra**: Matrices, tensors, decompositions (SVD, QR, LU, Cholesky, eigen)
- **Calculus**: Derivatives, integrals, gradients, Jacobians, Hessians
- **Probability**: Distributions (50+), sampling, MCMC, statistical tests
- **Information Theory**: Entropy, KL divergence, mutual information, channel capacity
- **Bayesian Methods**: Posterior inference, conjugate priors, Bayesian optimization
- **Numerical Methods**: Root finding, optimization, integration, ODE solvers
- **Graph Theory**: Graph algorithms, path finding, clustering, centrality
- **Discrete Math**: Combinatorics, set theory, logic, number theory

### 🖥️ Deployment & Visualization
- **Inference Server**: Async HTTP server with streaming support
- **Request Batching**: Continuous batching, dynamic batch sizing, priority scheduling
- **Caching**: Response cache (LRU), semantic cache, prefix KV cache sharing
- **Load Balancing**: Round-robin, least-connections, weighted strategies
- **Monitoring**: Metrics collection, alerting, health checking, Prometheus export
- **Rate Limiting**: Token bucket, sliding window, adaptive
- **Training Dashboard**: Loss curves, throughput tracking, gradient monitoring
- **Model Visualizer**: Architecture trees, parameter tables, FLOP estimation
- **Attention Visualizer**: Pattern analysis, head role identification, entropy

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Nexus LLM                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Deployment Layer                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Server  │ │ Batching │ │  Cache   │ │Load Bal  │ │Monitoring│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Inference Layer                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Quantize │ │ KV Cache │ │Speculative│ │Streaming │ │Distributed│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Alignment Layer                                                     │
│  ┌──────┐ ┌───────┐ ┌──────┐ ┌────────┐ ┌──────────┐               │
│  │ SFT  │ │  DPO  │ │ PPO  │ │ Reward │ │Const. AI │               │
│  └──────┘ └───────┘ └──────┘ └────────┘ └──────────┘               │
├─────────────────────────────────────────────────────────────────────┤
│  Agent & Reasoning Layer                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │CoT/ToT   │ │ Planning │ │ Tool Use │ │ Multi-Ag │ │ Memory   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Multimodal Layer                                                   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐              │
│  │Vision│ │Audio │ │Video │ │  Fusion  │ │Projector │              │
│  └──────┘ └──────┘ └──────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  Knowledge Layer                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Retrieval │ │  RAG     │ │   KG     │ │ Vector   │ │ Chunking │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Core Model                                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐│
│  │Transformer│ │Attention │ │ FFN  │ │MoE   │ │SSM   │ │Embeddings││
│  │  v1 + v2 │ │ variants │ │vari. │ │256exp│ │S4/Ma │ │ RoPE/ALi││
│  └──────────┘ └──────────┘ └──────┘ └──────┘ └──────┘ └──────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Training Layer                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Optimizers│ │Schedulers│ │  Parallel │ │  Converg. │ │  Data    │ │
│  │    x9    │ │   x11    │ │ 3D/ZeRO  │ │ Analysis  │ │ Strategy │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  Math Foundations                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Autodiff  │ │  Linear  │ │  Calculus│ │Probability│ │InfoTheory│ │
│  │          │ │  Algebra │ │          │ │          │ │          │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
nexus-llm/
├── src/nexus/
│   ├── __init__.py
│   ├── model/                     # Model architectures
│   │   ├── config.py              # Model configuration
│   │   ├── config_v2.py           # Extended configuration
│   │   ├── transformer.py         # Base transformer
│   │   ├── transformer_v2.py      # Enhanced transformer
│   │   ├── attention.py           # Multi-head attention
│   │   ├── attention_v2.py        # Attention variants (GQA, MQA, MLA)
│   │   ├── flash_attention.py     # Flash Attention v2/v3
│   │   ├── sparse_attention.py    # Sparse attention patterns
│   │   ├── ffn.py                 # Feed-forward networks
│   │   ├── ffn_v2.py             # FFN variants (SwiGLU, GeGLU)
│   │   ├── mixture_of_experts.py  # MoE with routing & parallelism
│   │   ├── state_space_models.py  # S4, Mamba, RWKV
│   │   ├── embeddings.py          # Token & position embeddings
│   │   ├── embeddings_v2.py       # Advanced embeddings
│   │   ├── rope.py                # Rotary Position Embedding
│   │   ├── positional.py          # All positional encodings
│   │   ├── norm.py                # Normalization layers
│   │   ├── normalization.py       # Advanced normalization
│   │   ├── activations.py         # 15+ activation functions
│   │   ├── residual.py            # Residual connection types
│   │   ├── innovations.py         # Architectural innovations
│   │   ├── output_head.py         # LM head
│   │   ├── multimodal_layers.py   # Multimodal model layers
│   │   ├── recurrent_layers.py    # Recurrent layer variants
│   │   ├── convolution_layers.py  # Convolutional layer variants
│   │   ├── regularization.py      # Dropout, drop path, stochastic depth
│   │   ├── model_utils.py         # Profiling, init, export utilities
│   │   └── __init__.py
│   │
│   ├── training/                  # Training infrastructure
│   │   ├── trainer.py             # Main trainer
│   │   ├── parallel.py            # 3D parallelism (TP/PP/DP)
│   │   ├── scheduler.py           # 11 LR schedulers
│   │   ├── optimizers.py          # 9 optimizers
│   │   ├── convergence.py         # Convergence analysis
│   │   ├── checkpoint.py          # Checkpoint management
│   │   ├── kernels.py             # Triton kernels
│   │   ├── lora.py                # LoRA/QLoRA/DoRA fine-tuning
│   │   ├── curriculum_learning.py # Curriculum training
│   │   ├── self_play.py           # Self-play data generation
│   │   ├── constitutional_ai.py    # Constitutional AI training
│   │   ├── hyperparameter_optimization.py  # HP search
│   │   ├── progressive_training.py # Progressive strategies
│   │   ├── data_strategy.py       # Data curation & mixing
│   │   └── training_utils.py     # Training utilities
│   │
│   ├── inference/                 # Inference & serving
│   │   ├── generator.py           # Text generation
│   │   ├── server.py              # Model serving
│   │   ├── quantize.py            # Quantization methods
│   │   ├── kv_cache.py            # KV cache management
│   │   ├── speculative.py         # Speculative decoding
│   │   ├── multi_token_pred.py    # Multi-token prediction
│   │   ├── distributed_serving.py # Distributed serving
│   │   ├── advanced_quantization.py # Advanced quantization
│   │   ├── inference_cache.py     # Inference caching
│   │   └── streaming_inference.py # Streaming generation
│   │
│   ├── alignment/                 # Alignment & RLHF
│   │   ├── sft.py                 # Supervised fine-tuning
│   │   ├── dpo.py                 # Direct preference optimization
│   │   ├── ppo.py                 # Proximal policy optimization
│   │   └── reward.py              # Reward modeling
│   │
│   ├── multimodal/                # Multimodal capabilities
│   │   ├── vision_encoder.py      # ViT, SigLIP, ConvNeXt
│   │   ├── audio_encoder.py       # Whisper, Mel spectrogram
│   │   ├── video_encoder.py       # TimeSformer, VideoSwin, ViViT
│   │   ├── cross_modal_fusion.py  # Fusion strategies
│   │   ├── multimodal_projector.py # Projectors (MLP, Q-Former)
│   │   ├── multimodal_config.py   # Multimodal configuration
│   │   └── multimodal_trainer.py  # Multimodal training
│   │
│   ├── reasoning/                 # Reasoning systems
│   │   ├── chain_of_thought.py    # CoT reasoning
│   │   ├── tree_of_thought.py     # Tree-of-thought
│   │   ├── planning.py            # Goal planning
│   │   ├── decomposition.py       # Task decomposition
│   │   ├── verification.py        # Answer verification
│   │   ├── self_consistency.py    # Self-consistency
│   │   ├── reasoning_config.py    # Reasoning configuration
│   │   ├── retrieval_augmented_reasoning.py  # RAG reasoning
│   │   └── reasoning_eval.py      # Reasoning evaluation
│   │
│   ├── agents/                    # Agent framework
│   │   ├── agent_framework.py     # Core agent
│   │   ├── agent_config.py        # Agent configuration
│   │   ├── tool_use.py            # Tool calling
│   │   ├── multi_agent.py         # Multi-agent systems
│   │   ├── planning_agent.py      # Planning agent
│   │   ├── memory_agent.py        # Memory agent
│   │   └── code_agent.py          # Code agent
│   │
│   ├── distributed/               # Distributed systems
│   │   ├── parallel.py            # Tensor/Pipeline/Data parallel
│   │   ├── rpc.py                 # RPC framework
│   │   ├── collective_ops.py      # Collective operations
│   │   ├── communication.py       # Communication utils
│   │   ├── fault_tolerance.py     # Fault tolerance
│   │   ├── elastic_training.py    # Elastic scaling
│   │   └── distributed_config.py  # Distributed configuration
│   │
│   ├── optimization/              # Model optimization
│   │   ├── quantization_advanced.py # GPTQ, AWQ, NF4, FP8
│   │   ├── pruning.py             # Structured/unstructured pruning
│   │   ├── distillation.py        # Knowledge distillation
│   │   ├── neural_architecture_search.py  # NAS
│   │   ├── model_compression.py   # Compression toolkit
│   │   ├── mixed_precision.py     # BF16/FP16 training
│   │   ├── compiler.py            # Model compilation
│   │   ├── gradient_checkpointing.py  # Memory optimization
│   │   ├── memory_optimization.py # Memory management
│   │   ├── inference_optimization.py  # Inference speed
│   │   └── optimization_config.py # Optimization config
│   │
│   ├── knowledge/                 # Knowledge & retrieval
│   │   ├── retrieval.py           # Dense/sparse/hybrid retrieval
│   │   ├── reranking.py           # Document reranking
│   │   ├── embeddings_advanced.py # Embedding models
│   │   ├── vector_store.py        # Vector stores (HNSW, IVF)
│   │   ├── knowledge_graph.py     # Knowledge graph
│   │   ├── rag_pipeline.py        # RAG pipeline
│   │   ├── document_processing.py # Document loaders
│   │   ├── chunking.py            # Text chunking strategies
│   │   └── knowledge_config.py    # Knowledge config
│   │
│   ├── memory/                    # Memory systems
│   │   ├── long_term_memory.py    # Persistent vector memory
│   │   ├── working_memory.py      # Working memory / scratchpad
│   │   ├── episodic_memory.py     # Experience memory
│   │   ├── semantic_memory.py     # Fact memory
│   │   ├── memory_manager.py      # Unified memory manager
│   │   └── memory_config.py       # Memory config
│   │
│   ├── tools_system/              # Tool system
│   │   ├── function_calling.py    # Function calling
│   │   ├── plugin_manager.py      # Plugin management
│   │   ├── builtin_tools.py       # Built-in tools
│   │   ├── tool_config.py         # Tool configuration
│   │   └── tool_sandbox.py        # Tool sandboxing
│   │
│   ├── visualization/             # Visualization
│   │   ├── training_dashboard.py  # Training monitoring
│   │   ├── model_visualizer.py    # Architecture visualization
│   │   ├── attention_visualizer.py # Attention analysis
│   │   └── data_visualizer.py     # Data analysis
│   │
│   ├── deployment/                # Deployment
│   │   ├── server_core.py         # Inference server
│   │   ├── batching.py            # Request batching
│   │   ├── caching.py             # Response caching
│   │   ├── load_balancer.py       # Load balancing
│   │   ├── monitoring.py          # Metrics & alerting
│   │   └── rate_limiter.py        # Rate limiting
│   │
│   ├── data/                      # Data processing
│   │   ├── tokenizer.py           # BPE + Unigram tokenizer
│   │   ├── dataset.py             # Dataset classes
│   │   ├── preprocessing.py       # Text preprocessing
│   │   ├── synthetic.py           # Synthetic data generation
│   │   ├── data_quality.py        # Quality filtering
│   │   ├── data_augmentation.py   # Data augmentation
│   │   ├── deduplication.py       # Deduplication (MinHash LSH)
│   │   ├── multilingual.py        # Multilingual support
│   │   ├── data_pipeline.py       # Data pipeline
│   │   └── web_scraping.py        # Web data collection
│   │
│   ├── evaluation/                # Evaluation
│   │   ├── benchmarks.py          # 20+ benchmarks
│   │   ├── comprehensive_benchmarks.py  # Extended benchmarks
│   │   ├── human_evaluation.py    # Human evaluation framework
│   │   ├── safety_evaluation.py   # Safety evaluation
│   │   └── bias_evaluation.py     # Bias evaluation
│   │
│   ├── safety/                    # Safety systems
│   │   ├── classifier.py          # Content classifier
│   │   ├── guardrails.py          # Input/output guardrails
│   │   ├── toxicity_detection.py  # Toxicity detection
│   │   ├── watermarking.py        # Output watermarking
│   │   ├── differential_privacy.py # Privacy-preserving training
│   │   └── secure_inference.py    # Secure inference
│   │
│   └── math/                      # Math foundations
│       ├── autodiff.py            # Automatic differentiation
│       ├── linalg.py              # Linear algebra
│       ├── tensor.py              # Tensor operations
│       ├── calculus.py            # Calculus
│       ├── numerical.py           # Numerical methods
│       ├── distributions.py       # Probability distributions
│       ├── sampling.py            # Sampling methods
│       ├── information.py         # Information theory
│       ├── bayesian.py            # Bayesian methods
│       ├── optimization_theory.py # Optimization theory
│       ├── graph_theory.py        # Graph theory
│       ├── probability_advanced.py # Advanced probability
│       ├── numerical_methods.py   # Numerical methods
│       └── discrete_math.py       # Discrete mathematics
│
├── configs/
│   └── base_100b.yaml             # 100B model configuration
│
├── scripts/
│   ├── train.py                   # Training entry point
│   ├── infer.py                   # Inference entry point
│   ├── serve.py                   # Serving entry point
│   ├── tokenize.py                # Tokenizer entry point
│   └── eval.py                    # Evaluation entry point
│
├── tests/
│   └── __init__.py
│
├── setup.py                       # Package setup
└── README.md                      # This file
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/nexus-llm.git
cd nexus-llm

# Install dependencies
pip install -e .

# Or install with training/inference extras
pip install -e ".[training,inference]"
```

### Training

```bash
# Train a 100B model with 3D parallelism
python -m nexus.scripts.train \\
    --config configs/base_100b.yaml \\
    --tensor_parallel 8 \\
    --pipeline_parallel 12 \\
    --data_path /data/train \\
    --output_dir /output/nexus-100b

# Fine-tune with LoRA
python -m nexus.scripts.train \\
    --model_path /output/nexus-100b \\
    --method lora \\
    --lora_rank 64 \\
    --data_path /data/instructions

# DPO alignment
python -m nexus.scripts.train \\
    --method dpo \\
    --model_path /output/nexus-100b \\
    --preference_data /data/preferences
```

### Inference

```bash
# Text generation
python -m nexus.scripts.infer \\
    --model_path /output/nexus-100b \\
    --prompt "Explain quantum computing in simple terms."

# Serve with OpenAI-compatible API
python -m nexus.scripts.serve \\
    --model_path /output/nexus-100b \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --tensor_parallel 4

# Quantized inference (4-bit NF4)
python -m nexus.scripts.infer \\
    --model_path /output/nexus-100b \\
    --quantize nf4 \\
    --bits 4
```

### Evaluation

```bash
# Run benchmarks
python -m nexus.scripts.eval \\
    --model_path /output/nexus-100b \\
    --benchmarks mmlu,hellaswag,arc,truthfulqa,gsm8k

# Safety evaluation
python -m nexus.scripts.eval \\
    --model_path /output/nexus-100b \\
    --safety_eval
```

---

## Model Configurations

| Config | Parameters | d_model | Heads | Layers | FFN Dim | Attention | Training |
|--------|-----------|---------|-------|--------|---------|-----------|----------|
| Small | 1.3B | 2048 | 16 | 24 | 8192 | GQA | 1 GPU |
| Base | 7B | 4096 | 32 | 32 | 16384 | GQA | 4 GPU |
| Large | 13B | 5120 | 40 | 40 | 20480 | GQA | 8 GPU |
| XL | 30B | 6656 | 52 | 48 | 26624 | GQA | 16 GPU |
| 100B | 100B+ | 12288 | 96 | 96 | 49152 | GQA | 96+ GPU |

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- sentencepiece >= 0.1.99
- tokenizers >= 0.15.0

Optional:
- flash-attn >= 2.5.0 (Flash Attention)
- deepspeed >= 0.13.0 (3D Parallelism)
- vllm >= 0.4.0 (Serving)
- wandb >= 0.16.0 (Logging)
- datasets >= 2.14.0 (Data loading)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ by the Nexus Community</strong>
</p>
