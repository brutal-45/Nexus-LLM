"""Vision Language Model Wrapper for Nexus-LLM.

Provides a high-level interface for vision-language models (VLMs),
enabling multimodal inference that combines images and text prompts.
Supports popular model families such as LLaVA, BLIP, and similar
architectures available through Hugging Face Transformers.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VLMBackend(str, Enum):
    """Supported VLM backends."""
    TRANSFORMERS = "transformers"
    LLAVA = "llava"
    BLIP = "blip"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VLMConfig:
    """Configuration for a vision-language model."""
    model_name: str = ""
    backend: VLMBackend = VLMBackend.TRANSFORMERS
    device: str = "auto"
    torch_dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    image_size: Tuple[int, int] = (336, 336)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend.value,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "trust_remote_code": self.trust_remote_code,
            "image_size": list(self.image_size),
        }


@dataclass
class VLMResponse:
    """Response from a vision-language model."""
    text: str = ""
    model_name: str = ""
    prompt: str = ""
    num_tokens_generated: int = 0
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "num_tokens_generated": self.num_tokens_generated,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


@dataclass
class MultimodalInput:
    """A multimodal input combining text and images."""
    text: str = ""
    images: List[Any] = field(default_factory=list)  # PIL Images or paths

    @classmethod
    def from_text(cls, text: str) -> "MultimodalInput":
        return cls(text=text, images=[])

    @classmethod
    def from_image_path(cls, path: str, prompt: str = "Describe this image.") -> "MultimodalInput":
        return cls(text=prompt, images=[path])

    @classmethod
    def from_image_paths(cls, paths: List[str], prompt: str = "Describe these images.") -> "MultimodalInput":
        return cls(text=prompt, images=list(paths))


# ---------------------------------------------------------------------------
# Vision Language Model
# ---------------------------------------------------------------------------

class VisionLanguageModel:
    """High-level wrapper for vision-language model inference.

    Provides a unified interface for loading and running multimodal
    models that accept both text and image inputs.  Automatically
    handles model and processor/tokenizer loading, image preprocessing,
    and text generation.

    Example::

        vlm = VisionLanguageModel(VLMConfig(model_name="llava-hf/llava-1.5-7b-hf"))
        vlm.load_model()

        response = vlm.generate(
            MultimodalInput(text="What is in this image?", images=["photo.jpg"])
        )
        print(response.text)

        vlm.unload_model()
    """

    def __init__(self, config: Optional[VLMConfig] = None) -> None:
        self._config = config or VLMConfig()
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._image_processor = None
        self._loaded = False

    @property
    def config(self) -> VLMConfig:
        """Return the model configuration."""
        return self._config

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Name of the configured model."""
        return self._config.model_name

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the vision-language model and processor into memory.

        Raises:
            ImportError: If transformers/torch is not installed.
            ValueError: If no model name is configured.
        """
        if self._loaded:
            return

        if not self._config.model_name:
            raise ValueError("No model name configured. Set VLMConfig.model_name.")

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "VLM loading requires transformers and torch: "
                "pip install transformers torch"
            )

        device = self._resolve_device()
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self._config.torch_dtype, torch.float16)

        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self._config.trust_remote_code,
        }
        if self._config.load_in_4bit:
            kwargs["load_in_4bit"] = True
        elif self._config.load_in_8bit:
            kwargs["load_in_8bit"] = True

        self._processor = AutoProcessor.from_pretrained(
            self._config.model_name,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._config.model_name, **kwargs
        )

        if not (self._config.load_in_4bit or self._config.load_in_8bit):
            self._model = self._model.to(device)
        self._model.eval()

        self._loaded = True

    def unload_model(self) -> None:
        """Release model and processor from memory."""
        import gc
        del self._model
        del self._processor
        del self._tokenizer
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolve_device(self) -> str:
        """Resolve the device string to a concrete device."""
        if self._config.device != "auto":
            return self._config.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        inputs: MultimodalInput,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> VLMResponse:
        """Generate text from a multimodal input.

        Args:
            inputs: MultimodalInput with text prompt and images.
            max_new_tokens: Override max tokens for this call.
            temperature: Override temperature for this call.
            top_p: Override top_p for this call.
            **kwargs: Additional generation kwargs.

        Returns:
            VLMResponse with generated text and metadata.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        import torch

        # Prepare images
        pil_images = self._prepare_images(inputs.images)

        # Prepare generation parameters
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self._config.max_new_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "top_p": top_p if top_p is not None else self._config.top_p,
            "top_k": kwargs.pop("top_k", self._config.top_k),
            "repetition_penalty": kwargs.pop("repetition_penalty", self._config.repetition_penalty),
            "do_sample": True,
        }
        gen_kwargs.update(kwargs)

        # Process inputs
        model_inputs = self._processor(
            text=inputs.text,
            images=pil_images if pil_images else None,
            return_tensors="pt",
        )

        device = next(self._model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(**model_inputs, **gen_kwargs)

        # Decode
        generated_text = self._processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        # Strip the prompt if the model echoes it
        if generated_text.startswith(inputs.text):
            generated_text = generated_text[len(inputs.text):].strip()

        return VLMResponse(
            text=generated_text,
            model_name=self._config.model_name,
            prompt=inputs.text,
            num_tokens_generated=len(output_ids[0]) - model_inputs["input_ids"].shape[1],
            finish_reason="stop",
            metadata={"num_images": len(pil_images)},
        )

    def batch_generate(
        self,
        inputs_list: List[MultimodalInput],
        **kwargs: Any,
    ) -> List[VLMResponse]:
        """Generate text for multiple multimodal inputs.

        Args:
            inputs_list: List of MultimodalInput objects.
            **kwargs: Additional generation kwargs.

        Returns:
            List of VLMResponse objects.
        """
        return [self.generate(inp, **kwargs) for inp in inputs_list]

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _prepare_images(self, images: List[Any]) -> List["Image.Image"]:
        """Convert image sources to PIL Image objects."""
        if not HAS_PIL:
            raise ImportError("Pillow is required for image handling")

        pil_images: List[Image.Image] = []
        for img_source in images:
            if isinstance(img_source, Image.Image):
                pil_images.append(img_source.convert("RGB"))
            elif isinstance(img_source, str):
                if os.path.isfile(img_source):
                    pil_images.append(Image.open(img_source).convert("RGB"))
                else:
                    raise FileNotFoundError(f"Image file not found: {img_source}")
            elif isinstance(img_source, bytes):
                pil_images.append(Image.open(io.BytesIO(img_source)).convert("RGB"))
            else:
                raise ValueError(f"Unsupported image source type: {type(img_source).__name__}")

        return pil_images

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Get model information and status.

        Returns:
            Dictionary with model name, loaded state, config, and device.
        """
        info: Dict[str, Any] = {
            "model_name": self._config.model_name,
            "backend": self._config.backend.value,
            "loaded": self._loaded,
            "config": self._config.to_dict(),
        }
        if self._loaded and self._model is not None:
            import torch
            info["device"] = str(next(self._model.parameters()).device)
            info["dtype"] = str(next(self._model.parameters()).dtype)
            param_count = sum(p.numel() for p in self._model.parameters())
            info["parameter_count"] = param_count
            info["parameter_count_human"] = f"{param_count / 1e9:.2f}B" if param_count >= 1e9 else f"{param_count / 1e6:.2f}M"
        return info
