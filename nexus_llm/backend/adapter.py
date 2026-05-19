"""Model adapters for Nexus-LLM backend.

Supports LoRA adapter loading/merging, adapter switching, and multi-adapter
support for serving multiple LoRA adapters simultaneously.
"""

import torch
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Supported adapter types."""
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    IA3 = "ia3"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"


class AdapterStatus(Enum):
    """Status of an adapter."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    MERGED = "merged"
    ERROR = "error"


@dataclass
class AdapterInfo:
    """Information about a loaded adapter."""
    adapter_id: str
    adapter_path: str
    adapter_type: AdapterType = AdapterType.LORA
    status: AdapterStatus = AdapterStatus.NOT_LOADED
    rank: int = 8
    alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    scaling: float = 1.0
    memory_mb: float = 0.0
    is_merged: bool = False


class AdapterManager:
    """Manages LoRA and other adapters for a base model.

    Supports loading, merging, switching between, and serving multiple
    adapters simultaneously.
    """

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        self._model = model
        self._tokenizer = tokenizer
        self._adapters: Dict[str, AdapterInfo] = {}
        self._active_adapter: Optional[str] = None
        self._merged_adapters: Dict[str, bool] = {}

    def load_adapter(
        self,
        adapter_path: str,
        adapter_id: Optional[str] = None,
        adapter_type: AdapterType = AdapterType.LORA,
        rank: int = 8,
        alpha: int = 16,
        target_modules: Optional[List[str]] = None,
    ) -> str:
        """Load a LoRA adapter from a path.

        Args:
            adapter_path: Path to the adapter weights.
            adapter_id: Unique identifier for the adapter. Defaults to directory name.
            adapter_type: Type of adapter to load.
            rank: LoRA rank (r parameter).
            alpha: LoRA alpha parameter.
            target_modules: Modules to apply the adapter to.

        Returns:
            The adapter ID.
        """
        if adapter_id is None:
            adapter_id = os.path.basename(adapter_path.rstrip("/"))

        if adapter_id in self._adapters:
            logger.warning(f"Adapter '{adapter_id}' already loaded")
            return adapter_id

        info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            adapter_type=adapter_type,
            status=AdapterStatus.LOADING,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules or ["q_proj", "v_proj"],
            scaling=alpha / rank,
        )

        try:
            self._load_lora_adapter(adapter_path, adapter_id, info)
            info.status = AdapterStatus.LOADED
            self._adapters[adapter_id] = info

            if self._active_adapter is None:
                self._active_adapter = adapter_id

            logger.info(f"Adapter '{adapter_id}' loaded from '{adapter_path}'")
            return adapter_id

        except Exception as e:
            info.status = AdapterStatus.ERROR
            logger.error(f"Failed to load adapter '{adapter_id}': {e}")
            raise

    def _load_lora_adapter(self, adapter_path: str, adapter_id: str, info: AdapterInfo) -> None:
        """Load a LoRA adapter using PEFT library."""
        try:
            from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

            if hasattr(self._model, "peft_config"):
                self._model.load_adapter(adapter_path, adapter_name=adapter_id)
            else:
                peft_config = LoraConfig(
                    r=info.rank,
                    lora_alpha=info.alpha,
                    target_modules=info.target_modules,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                try:
                    self._model = PeftModel.from_pretrained(
                        self._model, adapter_path, adapter_name=adapter_id, config=peft_config
                    )
                except Exception:
                    self._model = PeftModel.from_pretrained(
                        self._model, adapter_path, adapter_name=adapter_id
                    )

                adapter_size = sum(
                    p.numel() * p.element_size()
                    for name, p in self._model.named_parameters()
                    if adapter_id in name
                )
                info.memory_mb = adapter_size / (1024 * 1024)

        except ImportError:
            logger.warning("PEFT library not available, loading adapter weights directly")
            self._load_adapter_weights_directly(adapter_path, adapter_id, info)

    def _load_adapter_weights_directly(self, adapter_path: str, adapter_id: str, info: AdapterInfo) -> None:
        """Load adapter weights without PEFT (direct state dict loading)."""
        import json

        config_path = os.path.join(adapter_path, "adapter_config.json")
        weights_path = os.path.join(adapter_path, "adapter_model.bin")
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            info.rank = config.get("r", info.rank)
            info.alpha = config.get("lora_alpha", info.alpha)
            info.target_modules = config.get("target_modules", info.target_modules)
            info.scaling = info.alpha / info.rank

        adapter_state_dict = {}
        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file
                adapter_state_dict = load_file(safetensors_path)
            except ImportError:
                logger.warning("safetensors library not available")
        elif os.path.exists(weights_path):
            adapter_state_dict = torch.load(weights_path, map_location="cpu")

        prefix = f"adapter_{adapter_id}."
        new_state_dict = {}
        for key, value in adapter_state_dict.items():
            new_key = prefix + key
            new_state_dict[new_key] = value

        total_size = sum(v.numel() * v.element_size() for v in new_state_dict.values())
        info.memory_mb = total_size / (1024 * 1024)

        logger.info(f"Loaded adapter weights directly: {len(new_state_dict)} tensors")

    def set_active_adapter(self, adapter_id: str) -> None:
        """Switch to a specific adapter for inference."""
        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter '{adapter_id}' is not loaded. Available: {list(self._adapters.keys())}")

        if hasattr(self._model, "set_adapter"):
            self._model.set_adapter(adapter_id)

        self._active_adapter = adapter_id
        logger.info(f"Switched to adapter '{adapter_id}'")

    def get_active_adapter(self) -> Optional[str]:
        """Get the ID of the currently active adapter."""
        return self._active_adapter

    def merge_adapter(self, adapter_id: str) -> None:
        """Merge a LoRA adapter into the base model weights.

        After merging, the adapter cannot be unloaded separately.
        """
        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter '{adapter_id}' is not loaded")

        info = self._adapters[adapter_id]
        if info.is_merged:
            logger.warning(f"Adapter '{adapter_id}' is already merged")
            return

        if hasattr(self._model, "merge_and_unload"):
            self._model.merge_and_unload()
            info.is_merged = True
            info.status = AdapterStatus.MERGED
            self._merged_adapters[adapter_id] = True
            logger.info(f"Adapter '{adapter_id}' merged into base model")
        else:
            logger.warning("Model does not support merge_and_unload, merging manually")
            self._merge_adapter_manually(adapter_id)

    def _merge_adapter_manually(self, adapter_id: str) -> None:
        """Manually merge LoRA weights: W = W + (A @ B) * scaling."""
        info = self._adapters[adapter_id]
        base_state_dict = self._model.state_dict()

        scaling = info.scaling
        merged_count = 0

        for name, param in self._model.named_parameters():
            for module_name in info.target_modules:
                lora_A_key = f"adapter_{adapter_id}.base_model.model.{module_name}.lora_A.weight"
                lora_B_key = f"adapter_{adapter_id}.base_model.model.{module_name}.lora_B.weight"

                if lora_A_key in base_state_dict and lora_B_key in base_state_dict:
                    lora_A = base_state_dict[lora_A_key]
                    lora_B = base_state_dict[lora_B_key]
                    delta = (lora_B @ lora_A) * scaling
                    param.data += delta
                    merged_count += 1

        info.is_merged = True
        info.status = AdapterStatus.MERGED
        self._merged_adapters[adapter_id] = True
        logger.info(f"Manually merged {merged_count} LoRA modules for adapter '{adapter_id}'")

    def unload_adapter(self, adapter_id: str) -> None:
        """Unload an adapter (only if not merged)."""
        if adapter_id not in self._adapters:
            logger.warning(f"Adapter '{adapter_id}' is not loaded")
            return

        info = self._adapters[adapter_id]
        if info.is_merged:
            raise ValueError(f"Cannot unload merged adapter '{adapter_id}'. Merged adapters cannot be separated.")

        if hasattr(self._model, "unload"):
            self._model.unload()

        del self._adapters[adapter_id]

        if self._active_adapter == adapter_id:
            self._active_adapter = next(iter(self._adapters), None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Adapter '{adapter_id}' unloaded")

    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded adapters with their info."""
        result = {}
        for aid, info in self._adapters.items():
            result[aid] = {
                "adapter_id": info.adapter_id,
                "adapter_type": info.adapter_type.value,
                "status": info.status.value,
                "rank": info.rank,
                "alpha": info.alpha,
                "scaling": info.scaling,
                "memory_mb": info.memory_mb,
                "is_merged": info.is_merged,
                "target_modules": info.target_modules,
            }
        return result

    def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get info about a specific adapter."""
        return self._adapters.get(adapter_id)

    def get_total_adapter_memory(self) -> float:
        """Total memory used by all adapters in MB."""
        return sum(info.memory_mb for info in self._adapters.values())

    def enable_adapters(self) -> None:
        """Enable all adapters (un-disabling them)."""
        if hasattr(self._model, "enable_adapter_layers"):
            self._model.enable_adapter_layers()

    def disable_adapters(self) -> None:
        """Disable all adapters (use base model only)."""
        if hasattr(self._model, "disable_adapter_layers"):
            self._model.disable_adapter_layers()

    def shutdown(self) -> None:
        """Unload all non-merged adapters."""
        to_unload = [aid for aid, info in self._adapters.items() if not info.is_merged]
        for aid in to_unload:
            self.unload_adapter(aid)
