"""Model export: HuggingFace format, ONNX, GGML, safe tensors, quantized export."""

import os
import json
import logging
import shutil
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    output_dir: str = "./exported_model"
    format: str = "huggingface"
    dtype: str = "float16"
    quantize: bool = False
    quantization_bits: int = 4
    quantization_group_size: int = 128
    export_optimizer: bool = False
    export_tokenizer: bool = True
    safe_serialization: bool = True
    max_shard_size: str = "10GB"
    onnx_opset: int = 14
    onnx_dynamic_axes: bool = True
    ggml_type: str = "q4_0"
    include_metadata: bool = True
    model_card: bool = True


class ModelExporter:
    """Exports models to various formats for deployment."""

    def __init__(self, model: nn.Module, config: Optional[ExportConfig] = None):
        self.model = model
        self.config = config or ExportConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def export(self, format: Optional[str] = None, **kwargs) -> str:
        """Export the model to the specified format.

        Args:
            format: Export format. One of: huggingface, onnx, ggml, safetensors, torchscript, quantized.
            **kwargs: Additional export arguments.

        Returns:
            Path to the exported model directory.
        """
        fmt = format or self.config.format
        fmt = fmt.lower()

        export_methods = {
            "huggingface": self.export_huggingface,
            "onnx": self.export_onnx,
            "ggml": self.export_ggml,
            "safetensors": self.export_safetensors,
            "torchscript": self.export_torchscript,
            "quantized": self.export_quantized,
            "pytorch": self.export_pytorch,
        }

        if fmt not in export_methods:
            raise ValueError(
                f"Unknown export format: {fmt}. "
                f"Supported formats: {list(export_methods.keys())}"
            )

        logger.info(f"Exporting model in {fmt} format to {self.config.output_dir}")
        output_path = export_methods[fmt](**kwargs)
        logger.info(f"Model exported successfully to: {output_path}")
        return output_path

    def export_huggingface(self, **kwargs) -> str:
        """Export model in HuggingFace Transformers format."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Try using HuggingFace's save_pretrained if available
        if hasattr(self.model, "save_pretrained"):
            save_kwargs = {
                "save_directory": output_dir,
                "safe_serialization": self.config.safe_serialization,
            }
            if self.config.max_shard_size:
                save_kwargs["max_shard_size"] = self.config.max_shard_size
            self.model.save_pretrained(**save_kwargs)
        else:
            self._save_pytorch_state_dict(output_dir)
            self._save_config(output_dir)

        if self.config.include_metadata:
            self._save_metadata(output_dir, "huggingface")

        if self.config.model_card:
            self._save_model_card(output_dir)

        return output_dir

    def export_onnx(self, **kwargs) -> str:
        """Export model to ONNX format."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        onnx_path = os.path.join(output_dir, "model.onnx")

        # Try using HuggingFace's export if available
        if hasattr(self.model, "config") and hasattr(self.model.config, "model_type"):
            try:
                self._export_onnx_hf(output_dir, onnx_path)
                return output_dir
            except Exception as e:
                logger.warning(f"HF ONNX export failed: {e}. Falling back to torch export.")

        # Fallback to PyTorch ONNX export
        self._export_onnx_torch(onnx_path, **kwargs)

        if self.config.include_metadata:
            self._save_metadata(output_dir, "onnx")

        return output_dir

    def _export_onnx_hf(self, output_dir: str, onnx_path: str):
        """Export using HuggingFace's optimum library."""
        try:
            from optimum.exporters.onnx import main_export
            main_export(
                model_name_or_path=None,
                output=output_dir,
                task="text-generation",
                opset=self.config.onnx_opset,
            )
        except ImportError:
            raise RuntimeError("optimum library not installed for ONNX export.")

    def _export_onnx_torch(self, onnx_path: str, **kwargs):
        """Export using PyTorch's native ONNX export."""
        self.model.eval()
        device = next(self.model.parameters()).device

        dummy_input_ids = torch.randint(0, 1000, (1, 64), device=device)
        dummy_attention_mask = torch.ones((1, 64), dtype=torch.long, device=device)

        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]

        dynamic_axes = None
        if self.config.onnx_dynamic_axes:
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            }

        try:
            if hasattr(self.model, "forward"):
                forward_kwargs = {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}

                torch.onnx.export(
                    self.model,
                    (dummy_input_ids,),
                    onnx_path,
                    export_params=True,
                    opset_version=self.config.onnx_opset,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    def export_ggml(self, **kwargs) -> str:
        """Export model to GGML format for llama.cpp compatibility."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        ggml_path = os.path.join(output_dir, f"model-{self.config.ggml_type}.gguf")

        try:
            self._convert_to_ggml(ggml_path)
        except Exception as e:
            logger.warning(f"GGML conversion failed: {e}. Saving as PyTorch state dict instead.")
            self._save_pytorch_state_dict(output_dir)

        if self.config.include_metadata:
            self._save_metadata(output_dir, "ggml")

        return output_dir

    def _convert_to_ggml(self, output_path: str):
        """Convert model weights to GGML format."""
        state_dict = self.model.state_dict()

        with open(output_path, "wb") as f:
            # GGML header
            magic = b"GGUF"
            f.write(magic)

            version = 3
            f.write(version.to_bytes(4, "little"))

            # Write number of tensors
            num_tensors = len(state_dict)
            f.write(num_tensors.to_bytes(8, "little"))

            for name, tensor in state_dict.items():
                name_bytes = name.encode("utf-8")
                f.write(len(name_bytes).to_bytes(8, "little"))
                f.write(name_bytes)

                n_dims = tensor.dim()
                f.write(n_dims.to_bytes(4, "little"))

                for dim in tensor.shape:
                    f.write(dim.to_bytes(8, "little"))

                dtype_map = {
                    torch.float32: 0,
                    torch.float16: 1,
                    torch.int32: 2,
                    torch.int8: 3,
                }
                dtype_val = dtype_map.get(tensor.dtype, 0)
                f.write(dtype_val.to_bytes(4, "little"))

                # Align to 32 bytes
                pos = f.tell()
                padding = (32 - (pos % 32)) % 32
                f.write(b"\x00" * padding)

                tensor_np = tensor.numpy()
                f.write(tensor_np.tobytes())

        logger.info(f"GGML model written to {output_path}")

    def export_safetensors(self, **kwargs) -> str:
        """Export model using safetensors format."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            from safetensors.torch import save_file
            state_dict = self.model.state_dict()
            tensors = {k: v.contiguous() for k, v in state_dict.items()}
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(tensors, save_path)
            logger.info(f"SafeTensors model saved to {save_path}")
        except ImportError:
            logger.warning("safetensors not installed. Falling back to PyTorch format.")
            self._save_pytorch_state_dict(output_dir)

        self._save_config(output_dir)

        if self.config.include_metadata:
            self._save_metadata(output_dir, "safetensors")

        return output_dir

    def export_torchscript(self, **kwargs) -> str:
        """Export model as TorchScript."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        device = next(self.model.parameters()).device

        dummy_input = torch.randint(0, 1000, (1, 64), device=device)

        try:
            scripted = torch.jit.trace(self.model, dummy_input)
            scripted.save(os.path.join(output_dir, "model.pt"))
            logger.info("TorchScript model saved.")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            self._save_pytorch_state_dict(output_dir)

        return output_dir

    def export_quantized(self, **kwargs) -> str:
        """Export a quantized version of the model."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.config.quantization_bits == 8:
            return self._export_8bit_quantized(output_dir)
        elif self.config.quantization_bits == 4:
            return self._export_4bit_quantized(output_dir)
        else:
            logger.warning(
                f"Unsupported quantization bits: {self.config.quantization_bits}. "
                f"Falling back to 8-bit."
            )
            return self._export_8bit_quantized(output_dir)

    def _export_8bit_quantized(self, output_dir: str) -> str:
        """Export model with 8-bit quantization."""
        try:
            import bitsandbytes as bnb

            state_dict = self.model.state_dict()
            quantized_state = {}

            for name, param in state_dict.items():
                if param.dtype == torch.float32 or param.dtype == torch.float16:
                    if param.numel() > 256:
                        quantized = bnb.nn.Int8Params(param.cpu(), requires_grad=False)
                        quantized_state[name] = quantized
                    else:
                        quantized_state[name] = param
                else:
                    quantized_state[name] = param

            save_path = os.path.join(output_dir, "model_8bit.pt")
            torch.save(quantized_state, save_path)
            logger.info(f"8-bit quantized model saved to {save_path}")

        except ImportError:
            logger.warning("bitsandbytes not available. Saving standard model with fp16 dtype.")
            self._save_pytorch_state_dict(output_dir, dtype=torch.float16)

        return output_dir

    def _export_4bit_quantized(self, output_dir: str) -> str:
        """Export model with 4-bit quantization."""
        try:
            import bitsandbytes as bnb

            state_dict = self.model.state_dict()
            quantized_state = {}

            for name, param in state_dict.items():
                if param.dtype == torch.float32 or param.dtype == torch.float16:
                    if param.numel() > 256:
                        quantized = bnb.nn.Params4bit(
                            param.cpu(),
                            requires_grad=False,
                            compress_statistics=True,
                            quant_type="nf4",
                        )
                        quantized_state[name] = quantized
                    else:
                        quantized_state[name] = param
                else:
                    quantized_state[name] = param

            save_path = os.path.join(output_dir, "model_4bit.pt")
            torch.save(quantized_state, save_path)
            logger.info(f"4-bit quantized model saved to {save_path}")

        except ImportError:
            logger.warning("bitsandbytes not available. Saving standard model with fp16 dtype.")
            self._save_pytorch_state_dict(output_dir, dtype=torch.float16)

        return output_dir

    def export_pytorch(self, **kwargs) -> str:
        """Export model as standard PyTorch state dict."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)
        self._save_pytorch_state_dict(output_dir, dtype=dtype)
        self._save_config(output_dir)

        return output_dir

    def _save_pytorch_state_dict(self, output_dir: str, dtype: Optional[torch.dtype] = None):
        """Save the model state dict."""
        state_dict = self.model.state_dict()
        if dtype is not None:
            state_dict = {
                k: v.to(dtype) if v.is_floating_point() else v
                for k, v in state_dict.items()
            }

        if self.config.safe_serialization:
            try:
                from safetensors.torch import save_file
                tensors = {k: v.contiguous() for k, v in state_dict.items()}
                save_file(tensors, os.path.join(output_dir, "model.safetensors"))
                return
            except ImportError:
                pass

        save_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(state_dict, save_path)
        logger.info(f"PyTorch state dict saved to {save_path}")

    def _save_config(self, output_dir: str):
        """Save model configuration."""
        config = {}
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "to_dict"):
                config = self.model.config.to_dict()
            elif hasattr(self.model.config, "__dict__"):
                config = dict(self.model.config.__dict__)

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _save_metadata(self, output_dir: str, export_format: str):
        """Save export metadata."""
        metadata = {
            "export_format": export_format,
            "model_class": self.model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": self.config.dtype,
            "quantized": self.config.quantize,
            "quantization_bits": self.config.quantization_bits if self.config.quantize else None,
        }
        meta_path = os.path.join(output_dir, "export_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_model_card(self, output_dir: str):
        """Save a basic model card."""
        card_content = f"""---
language: en
library_name: nexus-llm
---

# Nexus-LLM Exported Model

This model was exported using Nexus-LLM's ModelExporter.

## Export Details

- **Format**: {self.config.format}
- **Dtype**: {self.config.dtype}
- **Quantized**: {self.config.quantize}
- **Parameters**: {sum(p.numel() for p in self.model.parameters()):,}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
```
"""
        card_path = os.path.join(output_dir, "README.md")
        with open(card_path, "w") as f:
            f.write(card_content)
