"""LoRA/PEFT fine-tuning: LoRA config, adapter training, merging, rank/alpha settings."""

import os
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    fan_in_fan_out: bool = False
    modules_to_save: Optional[List[str]] = None
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None
    rank_pattern: Optional[Dict[str, int]] = None
    alpha_pattern: Optional[Dict[str, int]] = None
    merge_weights: bool = True
    use_rslora: bool = False
    use_dora: bool = False

    def get_scaling(self, layer_name: Optional[str] = None) -> float:
        """Compute the LoRA scaling factor."""
        alpha = self.lora_alpha
        if self.alpha_pattern and layer_name and layer_name in self.alpha_pattern:
            alpha = self.alpha_pattern[layer_name]
        rank = self.r
        if self.rank_pattern and layer_name and layer_name in self.rank_pattern:
            rank = self.rank_pattern[layer_name]

        if self.use_rslora:
            return alpha / math.sqrt(rank)
        return alpha / rank


class LoRALayer(nn.Module):
    """LoRA adaptation layer that adds low-rank matrices to an existing linear layer."""

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        scaling: float = 1.0,
        use_dora: bool = False,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = scaling
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.fan_in_fan_out = fan_in_fan_out
        self.use_dora = use_dora
        self.merged = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros((in_features, r)))
        self.lora_B = nn.Parameter(torch.zeros((r, out_features)))

        if self.use_dora:
            self.lora_magnitude = nn.Parameter(torch.ones(out_features))

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge LoRA weights into the original layer."""
        if self.merged:
            return
        delta_weight = self._compute_delta()
        self.original_layer.weight.data += delta_weight
        if self.use_dora:
            self._merge_dora()
        self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from the original layer."""
        if not self.merged:
            return
        delta_weight = self._compute_delta()
        self.original_layer.weight.data -= delta_weight
        self.merged = False

    def _compute_delta(self) -> torch.Tensor:
        """Compute the delta weight from LoRA matrices."""
        result = self.lora_B @ self.lora_A.T
        result = result * self.scaling
        if self.fan_in_fan_out:
            result = result.T
        return result

    def _merge_dora(self):
        """Merge DORA magnitude into the weight."""
        weight_norm = self.original_layer.weight.data.norm(dim=1, keepdim=True)
        self.original_layer.weight.data = (
            self.lora_magnitude.unsqueeze(1) * self.original_layer.weight.data / weight_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)

        result = self.original_layer(x)

        if self.r > 0:
            lora_input = self.lora_dropout(x)
            lora_output = lora_input @ self.lora_A @ self.lora_B * self.scaling
            if self.use_dora:
                weight_norm = self.original_layer.weight.data.norm(dim=1)
                lora_output = (
                    self.lora_magnitude * (result + lora_output).norm(dim=-1) / weight_norm
                    * (result + lora_output)
                )
                result = lora_output
            else:
                result = result + lora_output

        return result


class FineTuner:
    """Manages LoRA/PEFT fine-tuning: applying adapters, training, and merging."""

    def __init__(self, model: nn.Module, config: Optional[LoRAConfig] = None):
        self.model = model
        self.config = config or LoRAConfig()
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.original_state: Dict[str, torch.Tensor] = {}

    def apply_lora(self) -> nn.Module:
        """Apply LoRA adapters to the model based on the config."""
        target_modules = set(self.config.target_modules)
        self._apply_lora_recursive(self.model, "", target_modules)

        trainable, total = self._count_parameters()
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total parameters "
            f"({100 * trainable / total:.2f}%)"
        )
        return self.model

    def _apply_lora_recursive(
        self,
        module: nn.Module,
        prefix: str,
        target_modules: set,
    ):
        """Recursively walk the model and inject LoRA layers."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                if self._is_target_module(name, full_name, target_modules):
                    scaling = self.config.get_scaling(full_name)
                    lora_layer = LoRALayer(
                        original_layer=child,
                        r=self.config.r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        fan_in_fan_out=self.config.fan_in_fan_out,
                        scaling=scaling,
                        use_dora=self.config.use_dora,
                    )
                    setattr(module, name, lora_layer)
                    self.lora_layers[full_name] = lora_layer
            else:
                self._apply_lora_recursive(child, full_name, target_modules)

    def _is_target_module(self, name: str, full_name: str, target_modules: set) -> bool:
        """Check if a module name matches any target module pattern."""
        for target in target_modules:
            if target in name or target in full_name:
                return True
        return False

    def _count_parameters(self) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def freeze_base_model(self):
        """Freeze all base model parameters, keeping only LoRA parameters trainable."""
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
                self.original_state[name] = param.data.clone()
        logger.info("Froze base model parameters; only LoRA adapters are trainable.")

    def unfreeze_base_model(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters.")

    def merge_adapters(self) -> nn.Module:
        """Merge all LoRA adapter weights into the base model."""
        merged_count = 0
        for name, lora_layer in self.lora_layers.items():
            if not lora_layer.merged:
                lora_layer.merge()
                merged_count += 1
        logger.info(f"Merged {merged_count} LoRA adapter(s) into base model.")
        return self.model

    def unmerge_adapters(self) -> nn.Module:
        """Unmerge all LoRA adapter weights from the base model."""
        for name, lora_layer in self.lora_layers.items():
            if lora_layer.merged:
                lora_layer.unmerge()
        logger.info("Unmerged all LoRA adapters from base model.")
        return self.model

    def save_adapter(self, output_dir: str):
        """Save only the LoRA adapter weights and config."""
        os.makedirs(output_dir, exist_ok=True)
        adapter_state = {}
        for name, lora_layer in self.lora_layers.items():
            adapter_state[f"{name}.lora_A"] = lora_layer.lora_A.data.cpu()
            adapter_state[f"{name}.lora_B"] = lora_layer.lora_B.data.cpu()
            if lora_layer.use_dora and hasattr(lora_layer, "lora_magnitude"):
                adapter_state[f"{name}.lora_magnitude"] = lora_layer.lora_magnitude.data.cpu()

        adapter_path = os.path.join(output_dir, "adapter_model.pt")
        torch.save(adapter_state, adapter_path)

        config_dict = {
            "r": self.config.r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "bias": self.config.bias,
            "task_type": self.config.task_type,
            "use_rslora": self.config.use_rslora,
            "use_dora": self.config.use_dora,
        }
        config_path = os.path.join(output_dir, "adapter_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved LoRA adapter to {output_dir}")

    def load_adapter(self, adapter_dir: str):
        """Load LoRA adapter weights from a directory."""
        adapter_path = os.path.join(adapter_dir, "adapter_model.pt")
        config_path = os.path.join(adapter_dir, "adapter_config.json")

        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        adapter_state = torch.load(adapter_path, map_location="cpu")
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A" in adapter_state:
                lora_layer.lora_A.data.copy_(adapter_state[f"{name}.lora_A"])
            if f"{name}.lora_B" in adapter_state:
                lora_layer.lora_B.data.copy_(adapter_state[f"{name}.lora_B"])
            if f"{name}.lora_magnitude" in adapter_state and hasattr(lora_layer, "lora_magnitude"):
                lora_layer.lora_magnitude.data.copy_(adapter_state[f"{name}.lora_magnitude"])

        logger.info(f"Loaded LoRA adapter from {adapter_dir}")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return only the trainable (LoRA) parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state dict containing only LoRA parameters."""
        lora_state = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                lora_state[name] = param.data.cpu()
        return lora_state
