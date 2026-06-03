"""Model management for Nexus-LLM.

Handles loading, unloading, and lifecycle management of HuggingFace models
with support for multiple precision modes and automatic device detection.
"""

import logging
import time
from enum import Enum
from typing import Dict, Optional, Any

import torch

from nexus_llm.core.exceptions import ModelLoadError, ModelNotFoundError
from nexus_llm.core.model_catalog import MODEL_CATALOG, get_model_info, ModelInfo

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    """States a managed model can be in."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class Precision(str, Enum):
    """Supported precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    EIGHTBIT = "8bit"
    FOURBIT = "4bit"


# ------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect the best available compute device.

    Returns:
        One of "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(device: str) -> str:
    """Resolve the device string, handling 'auto'."""
    if device == "auto":
        return detect_device()
    return device


# ------------------------------------------------------------------
# Precision helpers
# ------------------------------------------------------------------

def _get_torch_dtype(precision: str) -> Optional[torch.dtype]:
    """Map a precision string to a torch dtype.

    Returns None for quantised modes (8bit / 4bit) because BitsAndBytes
    handles the dtype internally.
    """
    mapping = {
        Precision.FP32: torch.float32,
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
    }
    return mapping.get(precision)


def _build_model_kwargs(
    precision: str,
    device: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the keyword arguments dict for ``from_pretrained``."""
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

    # Quantisation via bitsandbytes
    if precision == Precision.EIGHTBIT:
        kwargs["load_in_8bit"] = True
        # device_map is required for quantised loading
        kwargs["device_map"] = "auto" if device != "cpu" else {"": "cpu"}
    elif precision == Precision.FOURBIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto" if device != "cpu" else {"": "cpu"}
    else:
        dtype = _get_torch_dtype(precision)
        if dtype is not None:
            kwargs["dtype"] = dtype
        kwargs["device_map"] = device if device in ("cuda", "mps") else None

    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return kwargs


# ------------------------------------------------------------------
# ModelManager
# ------------------------------------------------------------------

class ModelManager:
    """Manages the lifecycle of a single HuggingFace model.

    Supports:
    * Causal LM (AutoModelForCausalLM) and Seq2Seq LM (AutoModelForSeq2SeqLM)
    * Automatic device detection (cuda / mps / cpu)
    * Multiple precision modes (fp32, fp16, bf16, 8bit, 4bit)
    * Memory usage reporting via ``psutil`` (graceful fallback)
    * State tracking (unloaded / loading / loaded / failed)
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._model_info: Optional[ModelInfo] = None
        self._state: ModelState = ModelState.UNLOADED
        self._device: Optional[str] = None
        self._precision: Optional[str] = None
        self._load_time: Optional[float] = None
        self._error_message: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> Optional[Any]:
        """The loaded HuggingFace model, or None."""
        return self._model

    @property
    def model_info(self) -> Optional[ModelInfo]:
        """Catalogue info for the currently loaded model."""
        return self._model_info

    @property
    def state(self) -> ModelState:
        """Current lifecycle state of the model."""
        return self._state

    @property
    def is_loaded(self) -> bool:
        """Whether a model is successfully loaded and ready."""
        return self._state == ModelState.LOADED and self._model is not None

    @property
    def device(self) -> Optional[str]:
        """The compute device the model is on (cuda / mps / cpu)."""
        return self._device

    @property
    def precision(self) -> Optional[str]:
        """The precision mode the model was loaded with."""
        return self._precision

    @property
    def model_id(self) -> Optional[str]:
        """Short model ID (e.g. "gpt2-medium") of the loaded model."""
        return self._model_info.id if self._model_info else None

    # ------------------------------------------------------------------
    # Load / Unload / Reload
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        device: str = "auto",
        precision: str = "fp32",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Load a model from the catalogue.

        Args:
            model_id: Short model ID from the catalogue.
            device: Compute device ("auto", "cuda", "mps", "cpu").
            precision: Precision mode (fp32, fp16, bf16, 8bit, 4bit).
            cache_dir: Optional HuggingFace cache directory.

        Raises:
            ModelNotFoundError: If *model_id* is not in the catalogue.
            ModelLoadError: If the model fails to load.
        """
        if self._state == ModelState.LOADING:
            raise ModelLoadError("A model is already being loaded. Wait for it to finish.")

        # Unload any existing model first
        if self.is_loaded or self._state == ModelState.FAILED:
            self.unload()

        # Validate model_id
        try:
            info = get_model_info(model_id)
        except ModelNotFoundError:
            raise

        self._state = ModelState.LOADING
        self._model_info = info
        logger.info(
            "Loading model %s (%s) — device=%s precision=%s",
            model_id, info.hf_id, device, precision,
        )

        resolved_device = _resolve_device(device)
        start_time = time.monotonic()

        try:
            kwargs = _build_model_kwargs(precision, resolved_device, cache_dir)
            hf_id = info.hf_id

            if info.model_type == "seq2seq":
                from transformers import AutoModelForSeq2SeqLM
                self._model = AutoModelForSeq2SeqLM.from_pretrained(hf_id, **kwargs)
            else:
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)

            # Move to device if not already handled by device_map / quantisation
            if (
                "device_map" not in kwargs or kwargs["device_map"] is None
            ) and resolved_device in ("cuda", "mps", "cpu"):
                self._model = self._model.to(resolved_device)

            self._model.eval()
            self._device = resolved_device
            self._precision = precision
            self._load_time = time.monotonic() - start_time
            self._state = ModelState.LOADED
            self._error_message = None

            logger.info(
                "Model %s loaded in %.2f s on %s (%s)",
                model_id, self._load_time, resolved_device, precision,
            )

        except Exception as exc:
            self._state = ModelState.FAILED
            self._error_message = str(exc)
            self._model = None
            logger.exception("Failed to load model %s", model_id)
            raise ModelLoadError(
                f"Failed to load model '{model_id}': {exc}"
            ) from exc

    def unload(self) -> None:
        """Unload the current model and free GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        model_name = self._model_info.id if self._model_info else "unknown"
        logger.info("Model %s unloaded", model_name)

        self._model_info = None
        self._state = ModelState.UNLOADED
        self._device = None
        self._precision = None
        self._load_time = None
        self._error_message = None

    def reload(
        self,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Reload the current model (optionally with new settings).

        Args:
            device: New device, or None to keep the current one.
            precision: New precision, or None to keep the current one.
            cache_dir: New cache dir, or None to keep the current one.

        Raises:
            ModelLoadError: If no model was previously loaded.
        """
        if self._model_info is None:
            raise ModelLoadError("No model to reload. Load a model first.")

        model_id = self._model_info.id
        new_device = device or self._device or "auto"
        new_precision = precision or self._precision or "fp32"

        self.unload()
        self.load(model_id, device=new_device, precision=new_precision, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Memory usage
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory usage information.

        Uses ``psutil`` for CPU/RAM stats and ``torch.cuda`` for GPU stats.
        Falls back gracefully if ``psutil`` is not installed.

        Returns:
            Dict with keys ``cpu``, ``ram``, ``gpu`` (if CUDA available),
            and ``model_device``.
        """
        result: Dict[str, Any] = {"model_device": self._device}

        # CPU/RAM via psutil (graceful fallback)
        try:
            import psutil
            proc = psutil.Process()
            mem_info = proc.memory_info()
            result["cpu"] = {
                "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            }
            result["ram"] = {
                "total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
                "used_percent": psutil.virtual_memory().percent,
            }
        except ImportError:
            result["cpu"] = {"note": "psutil not installed"}
            result["ram"] = {"note": "psutil not installed"}
        except Exception as exc:
            result["cpu"] = {"error": str(exc)}
            result["ram"] = {"error": str(exc)}

        # GPU via torch.cuda
        if torch.cuda.is_available():
            try:
                result["gpu"] = {
                    "device_name": torch.cuda.get_device_name(0),
                    "total_mb": round(torch.cuda.get_device_properties(0).total_mem / (1024 * 1024), 2),
                    "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
                }
            except Exception as exc:
                result["gpu"] = {"error": str(exc)}

        return result

    # ------------------------------------------------------------------
    # Info / summary
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Return a comprehensive summary dict of the manager state."""
        info: Dict[str, Any] = {
            "state": self._state.value,
            "model_id": self.model_id,
            "device": self._device,
            "precision": self._precision,
            "load_time_s": self._load_time,
            "error": self._error_message,
        }
        if self._model_info:
            info["model_info"] = {
                "name": self._model_info.name,
                "hf_id": self._model_info.hf_id,
                "category": self._model_info.category,
                "size": self._model_info.size,
                "params": self._model_info.params,
                "model_type": self._model_info.model_type,
                "recommended": self._model_info.recommended,
                "min_ram_gb": self._model_info.min_ram_gb,
            }
        if self.is_loaded and self._model is not None:
            info["model_dtype"] = str(getattr(self._model, "dtype", "unknown"))
            info["model_device"] = str(getattr(self._model, "device", "unknown"))
            # Parameter count
            try:
                total_params = sum(p.numel() for p in self._model.parameters())
                info["total_parameters"] = total_params
            except Exception:
                pass
        return info
