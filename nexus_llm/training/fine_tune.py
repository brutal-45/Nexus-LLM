"""LoRA / PEFT fine-tuning pipeline for Nexus-LLM.

Provides utilities for creating LoRA configurations, applying them to models,
preparing models for quantised training, and merging / saving adapters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nexus_llm.core.config import TrainingSettings
from nexus_llm.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default target modules per model family
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_MODULES: Dict[str, List[str]] = {
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "phi": ["q_proj", "v_proj", "k_proj", "dense"],
    "gpt2": ["c_attn", "c_proj"],
    "opt": ["q_proj", "v_proj"],
    "pythia": ["query_key_value", "dense"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bloom": ["query_key_value"],
    "mamba": ["in_proj", "out_proj"],
    "default": ["q_proj", "v_proj"],
}


def _infer_target_modules(model: Any) -> List[str]:
    """Heuristically determine LoRA target modules from model architecture."""
    name = type(model).__name__.lower()
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "").lower() if config else ""

    # Map common model_type values
    for key in _DEFAULT_TARGET_MODULES:
        if key in model_type or key in name:
            return _DEFAULT_TARGET_MODULES[key]

    # Fallback: inspect actual module names
    module_names = {name for name, _ in model.named_modules()}
    for key, targets in _DEFAULT_TARGET_MODULES.items():
        if any(t in " ".join(module_names) for t in targets):
            return targets

    return _DEFAULT_TARGET_MODULES["default"]


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------

class FineTuner:
    """Manages LoRA / PEFT fine-tuning lifecycle.

    Typical workflow::

        ft = FineTuner(training_settings)
        model = ft.apply_lora(model)
        model = ft.prepare_for_training(model)
        # ... train ...
        model = ft.merge_and_unload(model)
        ft.save_adapter(model, "./my-adapter")
    """

    def __init__(self, settings: Optional[TrainingSettings] = None) -> None:
        self._settings = settings or TrainingSettings()
        self._lora_config: Any = None
        self._is_applied = False

    @property
    def lora_config(self) -> Any:
        """The current LoRA configuration."""
        return self._lora_config

    @property
    def is_applied(self) -> bool:
        """Whether LoRA has been applied to a model."""
        return self._is_applied

    # ------------------------------------------------------------------
    # LoRA configuration
    # ------------------------------------------------------------------

    def create_lora_config(
        self,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        model: Optional[Any] = None,
    ) -> Any:
        """Create a LoraConfig with the given or default parameters.

        Args:
            r: LoRA rank. Defaults to ``settings.lora_r``.
            alpha: LoRA alpha. Defaults to ``settings.lora_alpha``.
            dropout: LoRA dropout. Defaults to ``settings.lora_dropout``.
            target_modules: Modules to apply LoRA to. Auto-inferred if None.
            bias: Bias handling ("none", "all", "lora_only").
            task_type: PEFT task type ("CAUSAL_LM" or "SEQ_2_SEQ_LM").
            model: Optional model for auto-inferring target modules.

        Returns:
            A ``peft.LoraConfig`` instance.
        """
        try:
            from peft import LoraConfig, TaskType
        except ImportError as exc:
            raise TrainingError(
                "The 'peft' package is required for LoRA fine-tuning. "
                "Install it with: pip install peft"
            ) from exc

        lora_r = r if r is not None else self._settings.lora_r
        lora_alpha = alpha if alpha is not None else self._settings.lora_alpha
        lora_dropout = dropout if dropout is not None else self._settings.lora_dropout

        if target_modules is None:
            if model is not None:
                target_modules = _infer_target_modules(model)
            else:
                target_modules = _DEFAULT_TARGET_MODULES["default"]

        task_type_enum = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

        self._lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=task_type_enum,
        )

        logger.info(
            "Created LoRA config: r=%d, alpha=%d, dropout=%.4f, targets=%s, task=%s",
            lora_r, lora_alpha, lora_dropout, target_modules, task_type,
        )
        return self._lora_config

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        model: Any,
        lora_config: Optional[Any] = None,
    ) -> Any:
        """Apply LoRA adapters to a model.

        Args:
            model: A HuggingFace ``PreTrainedModel``.
            lora_config: A ``LoraConfig``. If None, one will be created
                         automatically using current settings.

        Returns:
            The wrapped PEFT model.
        """
        try:
            from peft import get_peft_model
        except ImportError as exc:
            raise TrainingError(
                "The 'peft' package is required for LoRA fine-tuning. "
                "Install it with: pip install peft"
            ) from exc

        if lora_config is None:
            lora_config = self._lora_config or self.create_lora_config(model=model)

        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied — trainable: %s / %s params (%.2f%%)",
            f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1),
        )

        self._is_applied = True
        return model

    # ------------------------------------------------------------------
    # Prepare for quantised training
    # ------------------------------------------------------------------

    def prepare_for_training(self, model: Any) -> Any:
        """Prepare a model for k-bit training (8-bit / 4-bit).

        Applies gradient checkpointing and casts layer-norm / output
        parameters to float32 for numerical stability.

        Args:
            model: A PEFT-wrapped model.

        Returns:
            The prepared model.
        """
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            logger.warning(
                "peft.prepare_model_for_kbit_training not available; "
                "skipping k-bit preparation."
            )
            return model

        # Only prepare if the base model is quantised
        base_model = getattr(model, "base_model", model)
        is_quantised = hasattr(base_model, "is_quantized") and base_model.is_quantized

        if is_quantised:
            logger.info("Preparing quantised model for k-bit training")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
        else:
            # Still enable gradient checkpointing for memory savings
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")

        # Cast layer-norm and last layer to float32 for stability
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "norm" in name.lower() or "ln" in name.lower():
                    param.data = param.data.to(torch.float32)
                if "lm_head" in name or "embed_out" in name:
                    param.data = param.data.to(torch.float32)

        model.train()
        return model

    # ------------------------------------------------------------------
    # Merge and unload
    # ------------------------------------------------------------------

    def merge_and_unload(self, model: Any) -> Any:
        """Merge LoRA weights into the base model and unload the adapter.

        Args:
            model: A PEFT-wrapped model.

        Returns:
            The merged base model with LoRA weights baked in.
        """
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise TrainingError("Cannot merge: 'peft' package not installed.") from exc

        if not isinstance(model, PeftModel):
            logger.warning("Model is not a PEFT model; nothing to merge.")
            return model

        logger.info("Merging LoRA weights into base model")
        model = model.merge_and_unload()
        self._is_applied = False
        return model

    # ------------------------------------------------------------------
    # Save / Load adapters
    # ------------------------------------------------------------------

    def save_adapter(self, model: Any, output_dir: str) -> None:
        """Save LoRA adapter weights and configuration.

        Args:
            model: A PEFT-wrapped model.
            output_dir: Directory to save the adapter to.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise TrainingError("Cannot save adapter: 'peft' not installed.") from exc

        if not isinstance(model, PeftModel):
            raise TrainingError("Model is not a PEFT model; cannot save adapter.")

        model.save_pretrained(str(out_path))
        logger.info("LoRA adapter saved to %s", out_path)

    def load_adapter(
        self,
        model: Any,
        adapter_dir: str,
    ) -> Any:
        """Load a LoRA adapter onto a base model.

        Args:
            model: A base HuggingFace model.
            adapter_dir: Directory containing the saved adapter.

        Returns:
            The PEFT-wrapped model with the loaded adapter.
        """
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise TrainingError("Cannot load adapter: 'peft' not installed.") from exc

        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            raise TrainingError(f"Adapter directory not found: {adapter_path}")

        model = PeftModel.from_pretrained(model, str(adapter_path))
        self._is_applied = True

        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA adapter loaded from %s — trainable: %s / %s params",
            adapter_path, f"{trainable:,}", f"{total:,}",
        )
        return model
