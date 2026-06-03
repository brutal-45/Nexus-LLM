"""Model Manager - Handles loading, caching, and management of LLM models."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the lifecycle of language models including loading, caching,
    device placement, and memory optimization.
    """

    SUPPORTED_MODEL_TYPES = {
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
    }

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        model_type: str = "causal",
        device: str = "auto",
        precision: str = "fp32",
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = self._resolve_device(device)
        self.precision = precision

        self._model = None
        self._tokenizer = None
        self._config = None
        self._is_loaded = False
        self._model_info: Dict[str, Any] = {}

    def _resolve_device(self, device: str) -> str:
        """Resolve the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _get_dtype(self) -> torch.dtype:
        """Get the torch dtype based on precision setting."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.precision, torch.float32)
        # BF16 requires Ampere+ GPU
        if dtype == torch.bfloat16 and self.device == "cuda":
            if not torch.cuda.is_bf16_supported():
                logger.warning("BF16 not supported on this GPU, falling back to FP16")
                dtype = torch.float16
        return dtype

    def load_model(self) -> None:
        """
        Load the model and tokenizer into memory.
        Handles quantization, device placement, and memory optimization.
        """
        if self._is_loaded:
            logger.info(f"Model '{self.model_name}' is already loaded.")
            return

        logger.info(f"Loading model '{self.model_name}' on {self.device}...")
        dtype = self._get_dtype()

        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=False,
                padding_side="left",
            )

            # Ensure pad token exists
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            # Load model config
            self._config = AutoConfig.from_pretrained(self.model_name)

            # Load model with appropriate settings
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": False,
            }

            # Handle quantization
            if self.precision == "8bit" and self.device == "cuda":
                model_kwargs["load_in_8bit"] = True
            elif self.precision == "4bit" and self.device == "cuda":
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["dtype"] = dtype

            # Select the right model class
            model_class = self.SUPPORTED_MODEL_TYPES.get(
                self.model_type, AutoModelForCausalLM
            )
            self._model = model_class.from_pretrained(**model_kwargs)

            # Move to device (skip if quantized - already placed)
            if self.precision not in ("8bit", "4bit"):
                self._model = self._model.to(self.device)

            self._model.eval()

            # Collect model info
            self._model_info = {
                "name": self.model_name,
                "type": self.model_type,
                "device": str(self.device),
                "precision": self.precision,
                "dtype": str(dtype),
                "num_parameters": sum(p.numel() for p in self._model.parameters()),
                "num_parameters_billions": round(
                    sum(p.numel() for p in self._model.parameters()) / 1e9, 2
                ),
                "vocab_size": self._tokenizer.vocab_size,
                "max_position_embeddings": getattr(
                    self._config, "max_position_embeddings", "N/A"
                ),
            }

            self._is_loaded = True
            logger.info(
                f"Model loaded successfully: {self._model_info['num_parameters_billions']}B parameters"
            )

        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            self._is_loaded = False
            raise

    def unload_model(self) -> None:
        """Unload the model from memory to free GPU RAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False

        # Force garbage collection and GPU cache clear
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory freed.")

    def reload_model(self) -> None:
        """Reload the model (useful after training or config changes)."""
        self.unload_model()
        self.load_model()

    @property
    def model(self):
        """Get the loaded model."""
        if not self._is_loaded:
            self.load_model()
        return self._model

    @property
    def tokenizer(self):
        """Get the loaded tokenizer."""
        if not self._is_loaded:
            self.load_model()
        return self._tokenizer

    @property
    def config(self):
        """Get the model config."""
        if self._config is None:
            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self._model_info.copy()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory = {"cpu_ram_mb": 0.0, "gpu_ram_mb": 0.0}

        if self.device == "cuda" and torch.cuda.is_available():
            memory["gpu_ram_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory["gpu_ram_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        import psutil
        process = psutil.Process(os.getpid())
        memory["cpu_ram_mb"] = process.memory_info().rss / 1024 / 1024

        return memory

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if self._tokenizer is None:
            self.load_model()
        return len(self._tokenizer.encode(text))

    def estimate_generation_time(self, num_tokens: int) -> float:
        """Estimate generation time in seconds based on model size."""
        # Rough estimates based on typical performance
        if self.device == "cuda":
            tokens_per_second = 50 + (1e9 / (self._model_info.get("num_parameters", 1e8))) * 10
        else:
            tokens_per_second = 5 + (1e9 / (self._model_info.get("num_parameters", 1e8))) * 0.5
        return num_tokens / tokens_per_second
