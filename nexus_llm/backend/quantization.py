"""Model quantization for Nexus-LLM backend.

Supports 4-bit and 8-bit quantization using bitsandbytes, GGML format
support, and quantization configuration management.
"""

import torch
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP4 = "fp4"
    NF4 = "nf4"
    GGML_Q4_0 = "q4_0"
    GGML_Q4_1 = "q4_1"
    GGML_Q5_0 = "q5_0"
    GGML_Q5_1 = "q5_1"
    GGML_Q8_0 = "q8_0"
    GPTQ = "gptq"
    AWQ = "awq"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quant_type: QuantizationType = QuantizationType.NONE
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_storage: str = "uint8"
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    ggml_type: str = "q4_0"
    compute_dtype: Optional[str] = None
    quant_method: Optional[str] = None

    def to_bitsandbytes_config(self) -> Any:
        """Convert to a bitsandbytes configuration object."""
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("bitsandbytes and transformers are required for quantization")

        compute_dtype_str = self.bnb_4bit_compute_dtype or self.compute_dtype or "bfloat16"
        compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)
        quant_storage_str = self.bnb_4bit_quant_storage or "uint8"
        quant_storage = getattr(torch, quant_storage_str, torch.uint8)

        if self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.llm_int8_threshold,
                llm_int8_skip_modules=self.llm_int8_skip_modules,
            )
        elif self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_storage=quant_storage,
            )
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        result = {}
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if isinstance(val, Enum):
                result[f] = val.value
            elif isinstance(val, list):
                result[f] = list(val)
            else:
                result[f] = val
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizationConfig":
        """Create config from dictionary."""
        if "quant_type" in d and isinstance(d["quant_type"], str):
            d["quant_type"] = QuantizationType(d["quant_type"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @property
    def bits(self) -> int:
        """Effective number of bits per parameter."""
        if self.load_in_8bit or self.quant_type == QuantizationType.INT8:
            return 8
        if self.load_in_4bit or self.quant_type in (QuantizationType.INT4, QuantizationType.FP4, QuantizationType.NF4):
            return 4
        if self.quant_type == QuantizationType.GPTQ:
            return self.gptq_bits
        if self.quant_type == QuantizationType.AWQ:
            return self.awq_bits
        ggml_bits = {"q4_0": 4, "q4_1": 4, "q5_0": 5, "q5_1": 5, "q8_0": 8}
        if self.quant_type.value in ggml_bits:
            return ggml_bits[self.quant_type.value]
        return 32

    @property
    def compression_ratio(self) -> float:
        """Compression ratio compared to float32."""
        return 32.0 / max(1, self.bits)

    @property
    def is_quantized(self) -> bool:
        """Whether quantization is enabled."""
        return self.quant_type != QuantizationType.NONE or self.load_in_4bit or self.load_in_8bit


class QuantizationManager:
    """Manages model quantization configurations and application."""

    _PRESETS: Dict[str, QuantizationConfig] = {
        "int8": QuantizationConfig(quant_type=QuantizationType.INT8, load_in_8bit=True),
        "int4_nf4": QuantizationConfig(
            quant_type=QuantizationType.NF4, load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        ),
        "int4_fp4": QuantizationConfig(
            quant_type=QuantizationType.FP4, load_in_4bit=True,
            bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype="float16",
        ),
        "gptq_4bit": QuantizationConfig(
            quant_type=QuantizationType.GPTQ,
            gptq_bits=4, gptq_group_size=128, gptq_desc_act=False,
        ),
        "awq_4bit": QuantizationConfig(
            quant_type=QuantizationType.AWQ,
            awq_bits=4, awq_group_size=128, awq_zero_point=True,
        ),
        "ggml_q4_0": QuantizationConfig(
            quant_type=QuantizationType.GGML_Q4_0, ggml_type="q4_0",
        ),
        "ggml_q8_0": QuantizationConfig(
            quant_type=QuantizationType.GGML_Q8_0, ggml_type="q8_0",
        ),
    }

    def __init__(self):
        self._configs: Dict[str, QuantizationConfig] = {}

    def get_preset(self, name: str) -> QuantizationConfig:
        """Get a quantization preset by name."""
        if name not in self._PRESETS:
            available = ", ".join(self._PRESETS.keys())
            raise ValueError(f"Unknown quantization preset '{name}'. Available: {available}")
        return self._PRESETS[name]

    def list_presets(self) -> List[str]:
        """List available quantization presets."""
        return list(self._PRESETS.keys())

    def create_config(self, name: str, config: QuantizationConfig) -> None:
        """Save a named quantization configuration."""
        self._configs[name] = config

    def get_config(self, name: str) -> Optional[QuantizationConfig]:
        """Retrieve a saved quantization configuration."""
        return self._configs.get(name)

    def apply_quantization(
        self,
        model: Any,
        config: QuantizationConfig,
    ) -> Any:
        """Apply post-training quantization to an already loaded model.

        For 4-bit/8-bit quantization during loading, use the config's
        to_bitsandbytes_config() method when calling model loading.
        """
        if not config.is_quantized:
            logger.info("No quantization configured, returning model unchanged")
            return model

        if config.quant_type in (QuantizationType.INT8, QuantizationType.INT4,
                                  QuantizationType.FP4, QuantizationType.NF4):
            logger.info(
                f"BitsAndBytes quantization ({config.quant_type.value}) is applied during "
                "model loading. Use config.to_bitsandbytes_config() when loading."
            )
            return model

        if config.quant_type == QuantizationType.GPTQ:
            return self._apply_gptq_quantization(model, config)
        elif config.quant_type == QuantizationType.AWQ:
            return self._apply_awq_quantization(model, config)
        else:
            logger.warning(f"Post-load quantization for {config.quant_type.value} is not supported")
            return model

    def _apply_gptq_quantization(self, model: Any, config: QuantizationConfig) -> Any:
        """Apply GPTQ quantization to a model."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

            quantize_config = BaseQuantizeConfig(
                bits=config.gptq_bits,
                group_size=config.gptq_group_size,
                desc_act=config.gptq_desc_act,
            )
            logger.info(f"GPTQ quantization configured: {config.gptq_bits}-bit, group_size={config.gptq_group_size}")
            return model
        except ImportError:
            logger.error("auto-gptq is required for GPTQ quantization")
            raise

    def _apply_awq_quantization(self, model: Any, config: QuantizationConfig) -> Any:
        """Apply AWQ quantization to a model."""
        try:
            from awq import AutoAWQForCausalLM

            logger.info(f"AWQ quantization configured: {config.awq_bits}-bit, group_size={config.awq_group_size}")
            return model
        except ImportError:
            logger.error("autoawq is required for AWQ quantization")
            raise

    def estimate_memory_savings(
        self,
        model_size_gb: float,
        config: QuantizationConfig,
    ) -> Dict[str, float]:
        """Estimate memory savings from quantization."""
        original_bits = 16  # assume float16 baseline
        quantized_bits = config.bits
        quantized_size = model_size_gb * (quantized_bits / original_bits)
        savings = model_size_gb - quantized_size

        return {
            "original_size_gb": model_size_gb,
            "quantized_size_gb": quantized_size,
            "savings_gb": savings,
            "compression_ratio": config.compression_ratio,
            "effective_bits": quantized_bits,
        }

    def detect_quantization(self, model_path: str) -> Optional[QuantizationConfig]:
        """Auto-detect quantization format from model path/config files."""
        config = QuantizationConfig()

        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            quant_config = getattr(hf_config, "quantization_config", None)
            if quant_config is not None:
                method = quant_config.get("quant_method", "")
                if method == "gptq":
                    config.quant_type = QuantizationType.GPTQ
                    config.gptq_bits = quant_config.get("bits", 4)
                    config.gptq_group_size = quant_config.get("group_size", 128)
                    config.gptq_desc_act = quant_config.get("desc_act", False)
                    config.quant_method = "gptq"
                elif method == "awq":
                    config.quant_type = QuantizationType.AWQ
                    config.awq_bits = quant_config.get("bits", 4)
                    config.awq_group_size = quant_config.get("group_size", 128)
                    config.awq_zero_point = quant_config.get("zero_point", True)
                    config.quant_method = "awq"
                elif "bitsandbytes" in method or method == "bitsandbytes":
                    load_4bit = quant_config.get("load_in_4bit", False)
                    load_8bit = quant_config.get("load_in_8bit", False)
                    if load_4bit:
                        config.load_in_4bit = True
                        config.quant_type = QuantizationType.NF4
                        config.bnb_4bit_quant_type = quant_config.get("bnb_4bit_quant_type", "nf4")
                    elif load_8bit:
                        config.load_in_8bit = True
                        config.quant_type = QuantizationType.INT8
                return config

        except Exception as e:
            logger.debug(f"Could not auto-detect quantization: {e}")

        return None
