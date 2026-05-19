"""Model loading for Nexus-LLM backend.

Supports HuggingFace loader, safetensors loader, sharded model loading,
and progress tracking during model download and loading.
"""

import os
import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LoadFormat(Enum):
    """Model weight format."""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGML = "ggml"
    GGUF = "gguf"
    AUTO = "auto"


@dataclass
class LoadProgress:
    """Progress information for model loading."""
    stage: str = "initializing"
    current_file: str = ""
    files_total: int = 0
    files_loaded: int = 0
    bytes_total: int = 0
    bytes_loaded: int = 0
    elapsed_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def progress_pct(self) -> float:
        """Progress as a percentage (0-100)."""
        if self.bytes_total <= 0:
            if self.files_total <= 0:
                return 0.0
            return (self.files_loaded / self.files_total) * 100
        return (self.bytes_loaded / self.bytes_total) * 100

    @property
    def speed_mb_s(self) -> float:
        """Loading speed in MB/s."""
        if self.elapsed_seconds <= 0:
            return 0.0
        mb_loaded = self.bytes_loaded / (1024 * 1024)
        return mb_loaded / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        speed = self.speed_mb_s
        if speed <= 0:
            return 0.0
        mb_remaining = (self.bytes_total - self.bytes_loaded) / (1024 * 1024)
        return mb_remaining / speed

    def update(self, bytes_loaded: int = 0, file_loaded: bool = False) -> None:
        """Update progress."""
        self.bytes_loaded += bytes_loaded
        self.elapsed_seconds = time.time() - self.start_time
        if file_loaded:
            self.files_loaded += 1


@dataclass
class ModelLoadConfig:
    """Configuration for model loading."""
    model_path: str = ""
    load_format: LoadFormat = LoadFormat.AUTO
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False
    revision: str = "main"
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    offload_folder: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None
    quantization_config: Optional[Any] = None
    device_map: Optional[Any] = None
    attn_implementation: Optional[str] = None
    use_flash_attention: bool = False
    parallelize: bool = False


class ProgressCallback:
    """Callback for tracking model loading progress."""

    def __init__(self, callback_fn: Optional[Callable[[LoadProgress], None]] = None):
        self._callback_fn = callback_fn
        self.progress = LoadProgress()
        self._history: List[Dict[str, Any]] = []

    def __call__(self, progress_info: Dict[str, Any]) -> None:
        """Called with progress updates during loading."""
        if "stage" in progress_info:
            self.progress.stage = progress_info["stage"]
        if "current_file" in progress_info:
            self.progress.current_file = progress_info["current_file"]
        if "files_total" in progress_info:
            self.progress.files_total = progress_info["files_total"]
        if "bytes_total" in progress_info:
            self.progress.bytes_total = progress_info["bytes_total"]

        bytes_delta = progress_info.get("bytes_loaded", 0)
        self.progress.update(bytes_loaded=bytes_delta, file_loaded=progress_info.get("file_loaded", False))

        if self._callback_fn:
            self._callback_fn(self.progress)

        self._history.append({
            "timestamp": time.time(),
            "progress_pct": self.progress.progress_pct,
            "stage": self.progress.stage,
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the progress history."""
        return list(self._history)


class ModelLoader:
    """Unified model loader supporting multiple formats and progress tracking."""

    def __init__(self, progress_callback: Optional[Callable[[LoadProgress], None]] = None):
        self._progress_callback = ProgressCallback(progress_callback)
        self._loaded_models: Dict[str, Any] = {}

    def load_huggingface(
        self,
        model_path: str,
        config: Optional[ModelLoadConfig] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a model from HuggingFace Hub or local path.

        Returns (model, tokenizer).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        if config is None:
            config = ModelLoadConfig(model_path=model_path)

        self._progress_callback({"stage": "downloading", "current_file": model_path})

        torch_dtype = getattr(torch, config.dtype, torch.float16)

        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "revision": config.revision,
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
            "use_safetensors": config.use_safetensors,
        }

        if config.device_map is not None:
            load_kwargs["device_map"] = config.device_map
        elif config.device != "cpu":
            load_kwargs["device_map"] = config.device

        if config.max_memory is not None:
            load_kwargs["max_memory"] = config.max_memory

        if config.offload_folder is not None:
            load_kwargs["offload_folder"] = config.offload_folder

        if config.quantization_config is not None:
            load_kwargs["quantization_config"] = config.quantization_config

        if config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = config.attn_implementation
        elif config.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        load_kwargs.update(kwargs)

        self._progress_callback({"stage": "loading_config"})
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
        )

        total_params = getattr(hf_config, "num_params", None)
        if total_params is None:
            if hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_hidden_layers"):
                total_params = hf_config.hidden_size * hf_config.hidden_size * 12 * hf_config.num_hidden_layers
        if total_params:
            bytes_per_param = 2 if config.dtype in ("float16", "bfloat16") else 4
            self._progress_callback({
                "stage": "loading_weights",
                "bytes_total": total_params * bytes_per_param,
            })

        self._progress_callback({"stage": "loading_model"})
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        model.eval()

        self._progress_callback({"stage": "loading_tokenizer"})
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._progress_callback({"stage": "complete", "file_loaded": True})

        model_id = f"{model_path}:{config.revision}"
        self._loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": hf_config,
            "load_time": time.time(),
        }

        logger.info(f"Model loaded from '{model_path}' (dtype={config.dtype}, device={config.device})")
        return model, tokenizer

    def load_safetensors(
        self,
        model_path: str,
        config: Optional[ModelLoadConfig] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a model specifically from safetensors format.

        Falls back to HuggingFace loader which handles safetensors natively.
        """
        if config is None:
            config = ModelLoadConfig(model_path=model_path, load_format=LoadFormat.SAFETENSORS)

        config.use_safetensors = True
        return self.load_huggingface(model_path, config, **kwargs)

    def load_sharded(
        self,
        model_path: str,
        config: Optional[ModelLoadConfig] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a sharded model (split across multiple files).

        HuggingFace's from_pretrained handles sharding automatically.
        This method adds progress tracking for individual shards.
        """
        import os

        if config is None:
            config = ModelLoadConfig(model_path=model_path)

        shard_pattern = "model-"
        index_file = os.path.join(model_path, "model.safetensors.index.json") if os.path.isdir(model_path) else None

        if index_file and os.path.exists(index_file):
            with open(index_file, "r") as f:
                index_data = json.load(f)
            weight_files = index_data.get("weight_map", {})
            unique_files = set(weight_files.values())
            self._progress_callback({
                "stage": "loading_shards",
                "files_total": len(unique_files),
            })
        else:
            pytorch_index = os.path.join(model_path, "pytorch_model.bin.index.json")
            if os.path.exists(pytorch_index):
                with open(pytorch_index, "r") as f:
                    index_data = json.load(f)
                weight_files = index_data.get("weight_map", {})
                unique_files = set(weight_files.values())
                self._progress_callback({
                    "stage": "loading_shards",
                    "files_total": len(unique_files),
                })

        return self.load_huggingface(model_path, config, **kwargs)

    def load_gguf(
        self,
        model_path: str,
        config: Optional[ModelLoadConfig] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a model in GGUF format using llama-cpp-python.

        Returns (llama_model, tokenizer_wrapper) where tokenizer_wrapper
        provides basic encode/decode via the llama model.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for GGUF loading")

        if config is None:
            config = ModelLoadConfig(model_path=model_path, load_format=LoadFormat.GGUF)

        self._progress_callback({"stage": "loading_gguf", "current_file": model_path})

        n_gpu_layers = kwargs.pop("n_gpu_layers", -1)
        n_ctx = kwargs.pop("n_ctx", 4096)

        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
            **kwargs,
        )

        class GGUFTokenizer:
            """Minimal tokenizer wrapper for GGUF models."""
            def __init__(self, llama_model):
                self._model = llama_model

            def encode(self, text, **kw):
                return self._model.tokenize(text.encode("utf-8"), add_bos=kw.get("add_special_tokens", True))

            def decode(self, tokens, **kw):
                if isinstance(tokens, list):
                    return self._model.detokenize(tokens).decode("utf-8", errors="replace")
                return ""

            @property
            def eos_token_id(self):
                return self._model.token_eos()

            @property
            def bos_token_id(self):
                return self._model.token_bos()

            @property
            def pad_token_id(self):
                return self.eos_token_id

        self._progress_callback({"stage": "complete", "file_loaded": True})
        logger.info(f"GGUF model loaded from '{model_path}'")
        return model, GGUFTokenizer(model)

    def detect_format(self, model_path: str) -> LoadFormat:
        """Auto-detect the model format from the path contents."""
        if model_path.endswith(".gguf") or model_path.endswith(".ggml"):
            return LoadFormat.GGUF if model_path.endswith(".gguf") else LoadFormat.GGML

        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            if any(f.endswith(".safetensors") for f in files):
                return LoadFormat.SAFETENSORS
            if any(f.endswith(".bin") or f.endswith(".pt") for f in files):
                return LoadFormat.PYTORCH

        return LoadFormat.AUTO

    def get_progress(self) -> LoadProgress:
        """Get current loading progress."""
        return self._progress_callback.progress

    def verify_checksum(self, model_path: str, expected_sha256: Optional[str] = None) -> Optional[str]:
        """Verify the SHA256 checksum of model files."""
        sha256 = hashlib.sha256()
        if os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            computed = sha256.hexdigest()
            if expected_sha256:
                return "match" if computed == expected_sha256 else "mismatch"
            return computed
        return None

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a model without fully loading it."""
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            info = {
                "model_type": getattr(config, "model_type", "unknown"),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                "num_attention_heads": getattr(config, "num_attention_heads", None),
                "vocab_size": getattr(config, "vocab_size", None),
                "max_position_embeddings": getattr(config, "max_position_embeddings", None),
                "torch_dtype": str(getattr(config, "torch_dtype", "float16")),
            }

            num_params = getattr(config, "num_params", None)
            if num_params is not None:
                info["num_parameters"] = num_params

            return info
        except Exception as e:
            return {"error": str(e)}
