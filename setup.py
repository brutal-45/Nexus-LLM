"""
Nexus - Open-Source 100B+ Parameter Large Language Model
===========================================================
A complete, production-grade LLM codebase built from scratch with PyTorch.
No external AI APIs required. Fully open-source.

Architecture: Decoder-only Transformer with RoPE, GQA, SwiGLU, RMSNorm
Training: 3D Parallelism (TP/PP/DP), FSDP/ZeRO-3, BF16 Mixed Precision
Alignment: SFT → DPO → RLHF pipeline
Inference: PagedAttention, INT4/INT8 Quantization, Continuous Batching 
"""

from setuptools import setup, find_packages

setup(
    name="nexus",
    version="1.0.0",
    description="Open-Source 100B+ Parameter Large Language Model",
    author="Nexus Community",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.15.0",
        "datasets>=2.14.0",
        "safetensors>=0.4.0",
        "transformers>=4.36.0",
        "flash-attn>=2.5.0",
        "vllm>=0.4.0",
        "accelerate>=0.25.0",
        "deepspeed>=0.13.0",
        "wandb>=0.16.0",
        "tensorboard>=2.15.0",
        "fastapi>=0.108.0",
        "uvicorn>=0.25.0",
        "pydantic>=2.5.0",
        "aiohttp>=3.9.0",
        "multiprocess>=0.70.0",
        "lm-eval>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "ruff>=0.1.0",
        ],
        "training": [
            "deepspeed>=0.13.0",
            "wandb>=0.16.0",
        ],
        "inference": [
            "vllm>=0.4.0",
            "fastapi>=0.108.0",
            "uvicorn>=0.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nexus-train=nexus.scripts.train:main",
            "nexus-infer=nexus.scripts.infer:main",
            "nexus-serve=nexus.scripts.serve:main",
            "nexus-tokenize=nexus.scripts.tokenize:main",
            "nexus-eval=nexus.scripts.eval:main",
        ],
    },
)
