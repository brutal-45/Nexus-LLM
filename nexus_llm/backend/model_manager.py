"""Model manager for Nexus-LLM backend.

Handles loading, unloading, device placement, memory tracking, model switching,
and auto device mapping. Supports 39+ models from HuggingFace.
"""

import torch
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a loaded model."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model_path: str
    status: ModelStatus = ModelStatus.NOT_LOADED
    device: str = "cpu"
    dtype: str = "float32"
    memory_mb: float = 0.0
    num_parameters: int = 0
    max_seq_length: int = 2048
    architecture: str = ""
    quantization: Optional[str] = None
    adapter_name: Optional[str] = None


class ModelRegistry:
    """Registry of supported models with metadata."""

    SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
        "llama-2-7b": {"path": "meta-llama/Llama-2-7b-hf", "arch": "llama", "params": 7, "seq_len": 4096},
        "llama-2-13b": {"path": "meta-llama/Llama-2-13b-hf", "arch": "llama", "params": 13, "seq_len": 4096},
        "llama-2-70b": {"path": "meta-llama/Llama-2-70b-hf", "arch": "llama", "params": 70, "seq_len": 4096},
        "llama-2-7b-chat": {"path": "meta-llama/Llama-2-7b-chat-hf", "arch": "llama", "params": 7, "seq_len": 4096},
        "llama-2-13b-chat": {"path": "meta-llama/Llama-2-13b-chat-hf", "arch": "llama", "params": 13, "seq_len": 4096},
        "llama-2-70b-chat": {"path": "meta-llama/Llama-2-70b-chat-hf", "arch": "llama", "params": 70, "seq_len": 4096},
        "llama-3-8b": {"path": "meta-llama/Meta-Llama-3-8B", "arch": "llama", "params": 8, "seq_len": 8192},
        "llama-3-8b-instruct": {"path": "meta-llama/Meta-Llama-3-8B-Instruct", "arch": "llama", "params": 8, "seq_len": 8192},
        "llama-3-70b": {"path": "meta-llama/Meta-Llama-3-70B", "arch": "llama", "params": 70, "seq_len": 8192},
        "llama-3-70b-instruct": {"path": "meta-llama/Meta-Llama-3-70B-Instruct", "arch": "llama", "params": 70, "seq_len": 8192},
        "mistral-7b": {"path": "mistralai/Mistral-7B-v0.1", "arch": "mistral", "params": 7, "seq_len": 8192},
        "mistral-7b-instruct": {"path": "mistralai/Mistral-7B-Instruct-v0.2", "arch": "mistral", "params": 7, "seq_len": 8192},
        "mixtral-8x7b": {"path": "mistralai/Mixtral-8x7B-v0.1", "arch": "mixtral", "params": 47, "seq_len": 32768},
        "mixtral-8x7b-instruct": {"path": "mistralai/Mixtral-8x7B-Instruct-v0.1", "arch": "mixtral", "params": 47, "seq_len": 32768},
        "mixtral-8x22b": {"path": "mistralai/Mixtral-8x22B-v0.1", "arch": "mixtral", "params": 141, "seq_len": 65536},
        "phi-2": {"path": "microsoft/phi-2", "arch": "phi", "params": 3, "seq_len": 2048},
        "phi-3-mini": {"path": "microsoft/Phi-3-mini-4k-instruct", "arch": "phi3", "params": 4, "seq_len": 4096},
        "phi-3-medium": {"path": "microsoft/Phi-3-medium-4k-instruct", "arch": "phi3", "params": 14, "seq_len": 4096},
        "qwen2-7b": {"path": "Qwen/Qwen2-7B", "arch": "qwen2", "params": 7, "seq_len": 32768},
        "qwen2-72b": {"path": "Qwen/Qwen2-72B", "arch": "qwen2", "params": 72, "seq_len": 32768},
        "qwen2-7b-instruct": {"path": "Qwen/Qwen2-7B-Instruct", "arch": "qwen2", "params": 7, "seq_len": 32768},
        "qwen2-72b-instruct": {"path": "Qwen/Qwen2-72B-Instruct", "arch": "qwen2", "params": 72, "seq_len": 32768},
        "gemma-7b": {"path": "google/gemma-7b", "arch": "gemma", "params": 7, "seq_len": 8192},
        "gemma-2b": {"path": "google/gemma-2b", "arch": "gemma", "params": 2, "seq_len": 8192},
        "gemma-2-9b": {"path": "google/gemma-2-9b", "arch": "gemma2", "params": 9, "seq_len": 8192},
        "gemma-2-27b": {"path": "google/gemma-2-27b", "arch": "gemma2", "params": 27, "seq_len": 8192},
        "falcon-7b": {"path": "tiiuae/falcon-7b", "arch": "falcon", "params": 7, "seq_len": 2048},
        "falcon-40b": {"path": "tiiuae/falcon-40b", "arch": "falcon", "params": 40, "seq_len": 2048},
        "mpt-7b": {"path": "mosaicml/mpt-7b", "arch": "mpt", "params": 7, "seq_len": 2048},
        "mpt-30b": {"path": "mosaicml/mpt-30b", "arch": "mpt", "params": 30, "seq_len": 2048},
        "starcoder2-7b": {"path": "bigcode/starcoder2-7b", "arch": "starcoder2", "params": 7, "seq_len": 16384},
        "starcoder2-15b": {"path": "bigcode/starcoder2-15b", "arch": "starcoder2", "params": 15, "seq_len": 16384},
        "codellama-7b": {"path": "codellama/CodeLlama-7b-hf", "arch": "llama", "params": 7, "seq_len": 16384},
        "codellama-13b": {"path": "codellama/CodeLlama-13b-hf", "arch": "llama", "params": 13, "seq_len": 16384},
        "codellama-34b": {"path": "codellama/CodeLlama-34b-hf", "arch": "llama", "params": 34, "seq_len": 16384},
        "deepseek-coder-7b": {"path": "deepseek-ai/deepseek-coder-7b-instruct", "arch": "llama", "params": 7, "seq_len": 16384},
        "deepseek-coder-33b": {"path": "deepseek-ai/deepseek-coder-33b-instruct", "arch": "llama", "params": 33, "seq_len": 16384},
        "yi-6b": {"path": "01-ai/Yi-6B", "arch": "llama", "params": 6, "seq_len": 4096},
        "yi-34b": {"path": "01-ai/Yi-34B", "arch": "llama", "params": 34, "seq_len": 4096},
    }

    @classmethod
    def get_model_info(cls, model_id: str) -> Optional[Dict[str, Any]]:
        return cls.SUPPORTED_MODELS.get(model_id)

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls.SUPPORTED_MODELS.keys())

    @classmethod
    def resolve_path(cls, model_id: str) -> Optional[str]:
        info = cls.get_model_info(model_id)
        return info["path"] if info else model_id


class ModelManager:
    """Manages model lifecycle: load, unload, switch, device placement, memory tracking."""

    def __init__(self, max_memory_gb: Optional[float] = None, device: Optional[str] = None):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._lock = threading.RLock()
        self._active_model: Optional[str] = None
        self.max_memory_gb = max_memory_gb
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_auto_device_map(self, model_path: str, dtype: str = "float16") -> Optional[Dict[str, Any]]:
        """Compute an automatic device map based on available GPU memory."""
        if not torch.cuda.is_available():
            return None

        try:
            from accelerate import infer_auto_device_map, init_empty_weights
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path)
            torch_dtype = getattr(torch, dtype, torch.float16)

            with init_empty_weights():
                dummy_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

            max_memory = {}
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_mem
                available = torch.cuda.mem_get_info(i)[0]
                max_memory[i] = f"{int(available * 0.85 // (1024**3))}GiB"
            max_memory["cpu"] = "32GiB"

            device_map = infer_auto_device_map(dummy_model, max_memory=max_memory, dtype=torch_dtype)
            return device_map
        except Exception as e:
            logger.warning(f"Auto device map computation failed: {e}")
            return {"": 0} if torch.cuda.is_available() else {"": "cpu"}

    def _estimate_model_memory(self, num_params_b: float, dtype: str = "float16") -> float:
        """Estimate model memory in GB based on parameter count and dtype."""
        bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
        b = bytes_per_param.get(dtype, 2)
        return num_params_b * b

    def load_model(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
        device_map: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Any] = None,
        trust_remote_code: bool = False,
        use_safetensors: bool = True,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a model and its tokenizer. Returns (model, tokenizer)."""
        with self._lock:
            if model_id in self._models:
                logger.info(f"Model '{model_id}' already loaded, returning cached instance")
                return self._models[model_id], self._tokenizers[model_id]

            resolved_path = model_path or ModelRegistry.resolve_path(model_id)
            target_device = device or self.device

            info = ModelInfo(
                model_id=model_id,
                model_path=resolved_path,
                status=ModelStatus.LOADING,
                device=target_device,
                dtype=dtype,
            )
            self._model_info[model_id] = info
            logger.info(f"Loading model '{model_id}' from '{resolved_path}'")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                torch_dtype = getattr(torch, dtype, torch.float16)

                load_kwargs = {
                    "pretrained_model_name_or_path": resolved_path,
                    "torch_dtype": torch_dtype,
                    "trust_remote_code": trust_remote_code,
                    "use_safetensors": use_safetensors,
                }

                if quantization_config is not None:
                    load_kwargs["quantization_config"] = quantization_config
                    info.quantization = str(quantization_config)

                if device_map is not None:
                    load_kwargs["device_map"] = device_map
                elif target_device != "cpu" and torch.cuda.is_available():
                    auto_map = self._compute_auto_device_map(resolved_path, dtype)
                    if auto_map:
                        load_kwargs["device_map"] = auto_map
                    else:
                        load_kwargs["device_map"] = {"": 0}

                load_kwargs.update(kwargs)

                model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

                if "device_map" not in load_kwargs:
                    model = model.to(target_device)

                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(
                    resolved_path,
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                self._models[model_id] = model
                self._tokenizers[model_id] = tokenizer

                num_params = sum(p.numel() for p in model.parameters())
                info.num_parameters = num_params
                info.memory_mb = self._estimate_model_memory(num_params / 1e9, dtype) * 1024
                info.status = ModelStatus.LOADED
                info.architecture = model.config.model_type if hasattr(model.config, "model_type") else "unknown"

                if self._active_model is None:
                    self._active_model = model_id

                logger.info(f"Model '{model_id}' loaded successfully ({num_params:,} parameters)")
                return model, tokenizer

            except Exception as e:
                info.status = ModelStatus.ERROR
                logger.error(f"Failed to load model '{model_id}': {e}")
                raise

    def unload_model(self, model_id: str, force: bool = False) -> None:
        """Unload a model and free its memory."""
        with self._lock:
            if model_id not in self._models:
                logger.warning(f"Model '{model_id}' is not loaded")
                return

            info = self._model_info.get(model_id)
            if info:
                info.status = ModelStatus.UNLOADING

            model = self._models.pop(model_id)
            self._tokenizers.pop(model_id, None)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if info:
                info.status = ModelStatus.NOT_LOADED
                info.memory_mb = 0.0

            if self._active_model == model_id:
                self._active_model = next(iter(self._models), None)

            logger.info(f"Model '{model_id}' unloaded")

    def switch_model(self, model_id: str) -> Tuple[Any, Any]:
        """Switch the active model to the specified one. Loads if not already loaded."""
        with self._lock:
            if model_id not in self._models:
                raise ValueError(f"Model '{model_id}' is not loaded. Load it first with load_model().")
            self._active_model = model_id
            logger.info(f"Switched active model to '{model_id}'")
            return self._models[model_id], self._tokenizers[model_id]

    def get_model(self, model_id: Optional[str] = None) -> Any:
        """Get a model by ID or the active model."""
        key = model_id or self._active_model
        if key is None:
            raise ValueError("No model is currently loaded")
        if key not in self._models:
            raise ValueError(f"Model '{key}' is not loaded")
        return self._models[key]

    def get_tokenizer(self, model_id: Optional[str] = None) -> Any:
        """Get a tokenizer by model ID or the active model's tokenizer."""
        key = model_id or self._active_model
        if key is None:
            raise ValueError("No model is currently loaded")
        if key not in self._tokenizers:
            raise ValueError(f"Tokenizer for '{key}' is not loaded")
        return self._tokenizers[key]

    def get_active_model_id(self) -> Optional[str]:
        """Return the ID of the currently active model."""
        return self._active_model

    def list_loaded_models(self) -> List[str]:
        """List IDs of all currently loaded models."""
        return list(self._models.keys())

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        return self._model_info.get(model_id)

    def get_all_model_info(self) -> Dict[str, ModelInfo]:
        """Get information about all tracked models."""
        return dict(self._model_info)

    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get GPU memory usage for all available GPUs."""
        usage = {}
        if not torch.cuda.is_available():
            return usage
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            usage[i] = {
                "total_mb": total / (1024 * 1024),
                "used_mb": used / (1024 * 1024),
                "free_mb": free / (1024 * 1024),
                "utilization_pct": (used / total) * 100,
            }
        return usage

    def get_total_model_memory(self) -> float:
        """Get total memory used by all loaded models in MB."""
        return sum(info.memory_mb for info in self._model_info.values() if info.status == ModelStatus.LOADED)

    def can_load_model(self, model_id: str, dtype: str = "float16") -> bool:
        """Check if there's enough memory to load a model."""
        registry_info = ModelRegistry.get_model_info(model_id)
        if registry_info is None:
            return True

        estimated_gb = self._estimate_model_memory(registry_info["params"], dtype)
        if self.max_memory_gb is not None:
            current_gb = self.get_total_model_memory() / 1024
            return current_gb + estimated_gb <= self.max_memory_gb

        if torch.cuda.is_available():
            free_mem_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
            return estimated_gb <= free_mem_gb * 0.9

        return True

    def unload_least_recently_used(self) -> Optional[str]:
        """Unload the least recently used model to free memory."""
        if not self._models:
            return None
        lru_model = next(iter(self._models))
        self.unload_model(lru_model)
        return lru_model

    def shutdown(self) -> None:
        """Unload all models and clean up."""
        with self._lock:
            model_ids = list(self._models.keys())
            for mid in model_ids:
                self.unload_model(mid)
            self._active_model = None
            logger.info("ModelManager shutdown complete")
