"""Model quantizer for Nexus-LLM.

Wraps PyTorch / HuggingFace quantisation APIs to provide a unified
interface for int8, int4, fp16, bf16, and GGUF (mock) quantization
with accuracy and size measurement utilities.
"""

import copy
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from nexus_llm.quantization.config import QuantConfig

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Per-method byte estimates (relative to fp32 baseline = 4 bytes/param)
# -----------------------------------------------------------------------
_BYTES_PER_PARAM: Dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "gguf": 0.5625,  # Q4_0 approximate
}


class Quantizer:
    """Quantize neural-network models to lower-precision formats.

    The Quantizer accepts any model object that follows the HuggingFace
    ``PreTrainedModel`` convention (exposes ``parameters()`` and
    ``state_dict()``), or a plain ``nn.Module``.  For GGUF export the
    implementation is a **mock** that estimates file sizes and marks the
    model metadata without producing an actual .gguf file.

    Usage::

        quantizer = Quantizer()
        config = QuantConfig(method="int4", group_size=128, sym=True)
        quantized_model = quantizer.quantize(model, config)
        metrics = quantizer.measure_accuracy(model, quantized_model, test_data)
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialise the Quantizer.

        Args:
            device: Target device string (e.g. ``"cuda:0"``, ``"cpu"``).
                When *None*, the device is inferred from the model.
        """
        self._device = device

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------

    def quantize(self, model: Any, config: QuantConfig) -> Any:
        """Quantize a model according to the given configuration.

        Supported methods:

        * **int8** – Dynamic 8-bit integer quantization via
          ``torch.quantization`` or BitsAndBytes.
        * **int4** – 4-bit quantization (group-wise) via BitsAndBytes
          ``load_in_4bit`` or custom logic.
        * **fp16** – Cast all floating-point parameters to
          ``torch.float16``.
        * **bf16** – Cast all floating-point parameters to
          ``torch.bfloat16``.
        * **gguf** – Mock GGUF export (metadata only; no file produced).

        Args:
            model: A PyTorch ``nn.Module`` or HuggingFace
                ``PreTrainedModel``.
            config: Quantization configuration.

        Returns:
            A new quantized model (deep-copied from the original).

        Raises:
            ValueError: If the quantization method is not supported.
            RuntimeError: If the required backend library is unavailable.
        """
        if config.method not in _BYTES_PER_PARAM:
            raise ValueError(
                f"Unsupported quantization method '{config.method}'. "
                f"Supported: {list(_BYTES_PER_PARAM.keys())}"
            )

        logger.info("Quantizing model with method=%s", config.method)

        # Work on a deep copy so the original model is untouched
        quantized = copy.deepcopy(model)

        if config.method in ("fp16", "bf16"):
            quantized = self._cast_dtype(quantized, config.method)
        elif config.method == "int8":
            quantized = self._quantize_int8(quantized, config)
        elif config.method == "int4":
            quantized = self._quantize_int4(quantized, config)
        elif config.method == "gguf":
            quantized = self._mock_gguf(quantized, config)

        # Tag the model so downstream code can inspect quantization metadata
        self._set_quant_metadata(quantized, config)

        logger.info(
            "Quantization complete — method=%s, estimated size=%.2f MB",
            config.method,
            self.get_model_size(quantized) / (1024 * 1024),
        )
        return quantized

    # ------------------------------------------------------------------
    # Accuracy & size measurement
    # ------------------------------------------------------------------

    def measure_accuracy(
        self,
        original: Any,
        quantized: Any,
        test_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Measure accuracy degradation after quantization.

        Runs both the original and quantized models on *test_data* and
        computes per-sample and aggregate metrics.

        The function is model-agnostic: if the model exposes a
        ``generate`` method it will be called; otherwise ``__call__``
        (forward) is used.

        Args:
            original: The original (un-quantized) model.
            quantized: The quantized model.
            test_data: List of input dicts.  Each dict should contain at
                least an ``"input_ids"`` key (tensor) or a ``"prompt"``
                key (str).  Additional keys are forwarded to the model.

        Returns:
            Dict with keys:

            * ``num_samples`` – number of test samples evaluated.
            * ``avg_output_diff`` – average output difference (0 = identical).
            * ``max_output_diff`` – worst-case output difference.
            * ``time_original_ms`` – total inference time for the original
              model in milliseconds.
            * ``time_quantized_ms`` – total inference time for the
              quantized model in milliseconds.
            * ``speedup`` – ratio of original / quantized time (>1 means
              quantized is faster).
        """
        import torch
        import numpy as np

        num_samples = len(test_data)
        if num_samples == 0:
            return {
                "num_samples": 0,
                "avg_output_diff": 0.0,
                "max_output_diff": 0.0,
                "time_original_ms": 0.0,
                "time_quantized_ms": 0.0,
                "speedup": 1.0,
            }

        diffs: List[float] = []
        time_original = 0.0
        time_quantized = 0.0

        original.eval()
        quantized.eval()

        with torch.no_grad():
            for sample in test_data:
                # Prepare inputs
                inputs = self._prepare_inputs(sample)

                # Original model
                t0 = time.perf_counter()
                out_orig = self._model_forward(original, inputs)
                time_original += (time.perf_counter() - t0) * 1000

                # Quantized model
                t0 = time.perf_counter()
                out_quant = self._model_forward(quantized, inputs)
                time_quantized += (time.perf_counter() - t0) * 1000

                # Compute difference
                diff = self._compute_output_diff(out_orig, out_quant)
                diffs.append(diff)

        avg_diff = float(np.mean(diffs)) if diffs else 0.0
        max_diff = float(np.max(diffs)) if diffs else 0.0
        speedup = (time_original / time_quantized) if time_quantized > 0 else 1.0

        result = {
            "num_samples": num_samples,
            "avg_output_diff": avg_diff,
            "max_output_diff": max_diff,
            "time_original_ms": round(time_original, 2),
            "time_quantized_ms": round(time_quantized, 2),
            "speedup": round(speedup, 4),
        }
        logger.info("Accuracy measurement: %s", result)
        return result

    def get_model_size(self, model: Any) -> int:
        """Return the estimated model size in bytes.

        For PyTorch models this iterates over ``state_dict()`` tensors;
        for objects with a ``get_memory_footprint`` method (HuggingFace)
        that value is used instead.  As a last resort the size is
        estimated from the number of parameters and quantization metadata.

        Args:
            model: A model object.

        Returns:
            Estimated size in bytes.
        """
        # HuggingFace convenience method
        if hasattr(model, "get_memory_footprint"):
            try:
                return int(model.get_memory_footprint())
            except Exception:
                pass

        # PyTorch state_dict
        try:
            import torch
            sd = model.state_dict()
            total = sum(v.nelement() * v.element_size() for v in sd.values() if isinstance(v, torch.Tensor))
            if total > 0:
                return total
        except Exception:
            pass

        # Estimate from parameter count + quantization metadata
        num_params = self._count_parameters(model)
        method = getattr(model, "_quant_method", "fp32")
        bytes_per_param = _BYTES_PER_PARAM.get(method, 4.0)
        return int(num_params * bytes_per_param)

    def compare_sizes(self, original: Any, quantized: Any) -> Dict[str, Any]:
        """Compare sizes of the original and quantized models.

        Args:
            original: The original model.
            quantized: The quantized model.

        Returns:
            Dict with keys:

            * ``original_bytes`` – size of the original model in bytes.
            * ``quantized_bytes`` – size of the quantized model in bytes.
            * ``reduction_bytes`` – absolute size reduction in bytes.
            * ``reduction_pct`` – size reduction as a percentage.
            * ``compression_ratio`` – original / quantized ratio.
        """
        orig_size = self.get_model_size(original)
        quant_size = self.get_model_size(quantized)
        reduction_bytes = orig_size - quant_size
        reduction_pct = (reduction_bytes / orig_size * 100) if orig_size > 0 else 0.0
        compression_ratio = (orig_size / quant_size) if quant_size > 0 else float("inf")

        return {
            "original_bytes": orig_size,
            "quantized_bytes": quant_size,
            "reduction_bytes": reduction_bytes,
            "reduction_pct": round(reduction_pct, 2),
            "compression_ratio": round(compression_ratio, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers – dtype casting
    # ------------------------------------------------------------------

    @staticmethod
    def _cast_dtype(model: Any, method: str) -> Any:
        """Cast model parameters to fp16 or bf16."""
        import torch

        dtype = torch.float16 if method == "fp16" else torch.bfloat16
        try:
            model = model.to(dtype)
            logger.debug("Casted model to %s", dtype)
        except Exception as exc:
            logger.warning("Failed to cast model to %s: %s", dtype, exc)
            # Attempt manual conversion for nn.Module
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if param.is_floating_point():
                        param.data = param.data.to(dtype)
        return model

    # ------------------------------------------------------------------
    # Internal helpers – int8
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_int8(model: Any, config: QuantConfig) -> Any:
        """Apply dynamic int8 quantization.

        Attempts BitsAndBytes first, then falls back to PyTorch native
        dynamic quantization on the CPU.
        """
        try:
            import bitsandbytes as bnb
            import torch

            def _replace_linear(module: torch.nn.Module) -> torch.nn.Module:
                """Replace Linear layers with 8-bit equivalents."""
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(
                            module,
                            name,
                            bnb.nn.Linear8bitLt(
                                child.in_features,
                                child.out_features,
                                has_bias=child.bias is not None,
                                threshold=6.0,
                            ),
                        )
                    else:
                        _replace_linear(child)
                return module

            _replace_linear(model)
            logger.debug("Applied BitsAndBytes int8 quantization")
            return model
        except ImportError:
            logger.debug("BitsAndBytes not available, using PyTorch dynamic quantization")

        try:
            import torch

            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            logger.debug("Applied PyTorch dynamic int8 quantization")
        except Exception as exc:
            logger.warning("int8 quantization failed: %s – returning unmodified model", exc)

        return model

    # ------------------------------------------------------------------
    # Internal helpers – int4
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_int4(model: Any, config: QuantConfig) -> Any:
        """Apply 4-bit group-wise quantization.

        Uses BitsAndBytes when available; otherwise performs a simulated
        4-bit quantization by rounding weights to 16 discrete levels
        within each group (for testing / benchmarking without GPUs).
        """
        try:
            import bitsandbytes as bnb
            import torch

            def _replace_linear(module: torch.nn.Module) -> torch.nn.Module:
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(
                            module,
                            name,
                            bnb.nn.Linear4bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                compute_dtype=torch.float16,
                                compress_statistics=not config.sym,
                                quant_type="nf4",
                                quant_storage=torch.uint8,
                            ),
                        )
                    else:
                        _replace_linear(child)
                return module

            _replace_linear(model)
            logger.debug("Applied BitsAndBytes int4 quantization (group_size=%d)", config.group_size)
            return model
        except ImportError:
            logger.debug("BitsAndBytes not available, using simulated int4 quantization")

        # Simulated 4-bit quantization for CPU / testing environments
        import torch
        import numpy as np

        def _simulated_int4(weight: torch.Tensor, group_size: int, sym: bool) -> torch.Tensor:
            """Quantize a weight tensor to simulated 4-bit precision."""
            flat = weight.data.flatten()
            num_groups = max(1, flat.numel() // group_size)
            pad_len = num_groups * group_size - flat.numel()
            if pad_len > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_len))
            grouped = flat.reshape(num_groups, group_size)

            if sym:
                # Symmetric: scale = max(|x|) / 7, levels [-7, 7]
                scale = grouped.abs().amax(dim=1, keepdim=True) / 7.0
                scale = scale.clamp(min=1e-8)
                quantized = torch.clamp(torch.round(grouped / scale), -7, 7)
                dequantized = quantized * scale
            else:
                # Asymmetric: min-max range mapped to [0, 15]
                g_min = grouped.amin(dim=1, keepdim=True)
                g_max = grouped.amax(dim=1, keepmin=True)  # typo-safe
                g_max = grouped.amax(dim=1, keepdim=True)
                scale = (g_max - g_min) / 15.0
                scale = scale.clamp(min=1e-8)
                zero_point = g_min
                quantized = torch.clamp(torch.round((grouped - zero_point) / scale), 0, 15)
                dequantized = quantized * scale + zero_point

            return dequantized.reshape_as(weight.data)

        if hasattr(model, "parameters"):
            for param in model.parameters():
                if param.is_floating_point() and param.numel() >= config.group_size:
                    param.data = _simulated_int4(param.data, config.group_size, config.sym)

        logger.debug("Applied simulated int4 quantization (group_size=%d, sym=%s)", config.group_size, config.sym)
        return model

    # ------------------------------------------------------------------
    # Internal helpers – GGUF mock
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_gguf(model: Any, config: QuantConfig) -> Any:
        """Mock GGUF quantization.

        Does not produce an actual GGUF file but tags the model so that
        size estimations and downstream tooling can recognise the
        format.
        """
        logger.info("GGUF quantization is a mock – no .gguf file will be produced")
        return model

    # ------------------------------------------------------------------
    # Internal helpers – utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _set_quant_metadata(model: Any, config: QuantConfig) -> None:
        """Attach quantization metadata to the model object."""
        model._quant_method = config.method  # type: ignore[attr-defined]
        model._quant_config = config.to_dict()  # type: ignore[attr-defined]

    @staticmethod
    def _count_parameters(model: Any) -> int:
        """Count total number of parameters in the model."""
        if hasattr(model, "num_parameters"):
            try:
                return int(model.num_parameters())
            except Exception:
                pass
        if hasattr(model, "parameters"):
            try:
                return sum(p.numel() for p in model.parameters())
            except Exception:
                pass
        return 0

    @staticmethod
    def _prepare_inputs(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw sample dict into model-ready inputs."""
        import torch

        inputs: Dict[str, Any] = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value
            elif isinstance(value, str) and key == "prompt":
                # Store the text prompt; callers that have a tokenizer
                # should pre-tokenise before passing to measure_accuracy.
                inputs["prompt"] = value
            else:
                inputs[key] = value
        return inputs

    @staticmethod
    def _model_forward(model: Any, inputs: Dict[str, Any]) -> Any:
        """Run a forward pass, preferring ``generate`` when available."""
        import torch

        if hasattr(model, "generate"):
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                return model.generate(input_ids=input_ids, max_new_tokens=32)

        # Fallback to forward / __call__
        forward_kwargs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "position_ids", "token_type_ids")
        }
        if forward_kwargs:
            return model(**forward_kwargs)
        return model(inputs)

    @staticmethod
    def _compute_output_diff(out_orig: Any, out_quant: Any) -> float:
        """Compute a scalar difference between two model outputs."""
        import torch
        import numpy as np

        def _to_numpy(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().float().numpy()
            if isinstance(obj, (tuple, list)) and len(obj) > 0:
                return _to_numpy(obj[0])
            return np.array(obj, dtype=np.float32)

        try:
            arr_orig = np.asarray(_to_numpy(out_orig)).flatten()
            arr_quant = np.asarray(_to_numpy(out_quant)).flatten()

            # Align lengths (e.g. generate may produce different token counts)
            min_len = min(len(arr_orig), len(arr_quant))
            arr_orig = arr_orig[:min_len]
            arr_quant = arr_quant[:min_len]

            denom = max(np.abs(arr_orig).max(), 1e-8)
            return float(np.mean(np.abs(arr_orig - arr_quant)) / denom)
        except Exception:
            return float("inf")
