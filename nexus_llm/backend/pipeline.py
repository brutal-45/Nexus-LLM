"""Pipeline management for Nexus-LLM backend.

Supports text-generation pipeline, text2text-generation pipeline, and
custom pipeline creation with configurable components.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .generation import GenerationConfig, GenerationPresets
from .tokenizer_utils import TokenizerWrapper

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Supported pipeline types."""
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    CONVERSATIONAL = "conversational"
    CUSTOM = "custom"


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""
    pipeline_type: PipelineType = PipelineType.TEXT_GENERATION
    model_id: str = ""
    device: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False
    revision: str = "main"
    use_safetensors: bool = True
    quantization_config: Optional[Any] = None
    generation_config: Optional[GenerationConfig] = None
    custom_preprocessor: Optional[Callable] = None
    custom_postprocessor: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f_name in self.__dataclass_fields__:
            val = getattr(self, f_name)
            if isinstance(val, Enum):
                result[f_name] = val.value
            elif val is not None and not callable(val):
                result[f_name] = val
        return result


class Pipeline:
    """Base pipeline class wrapping a model and tokenizer for inference."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: PipelineConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
        self._config = config
        self._generation_config = config.generation_config or GenerationPresets.balanced()

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> TokenizerWrapper:
        return self._tokenizer

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @generation_config.setter
    def generation_config(self, config: GenerationConfig) -> None:
        self._generation_config = config

    def preprocess(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Preprocess inputs into model-ready format."""
        if self._config.custom_preprocessor:
            return self._config.custom_preprocessor(inputs, **kwargs)

        if isinstance(inputs, str):
            encoded = self._tokenizer.encode_with_padding(
                [inputs],
                return_tensors="pt",
                **kwargs,
            )
            return {k: v for k, v in encoded.items()}
        elif isinstance(inputs, list):
            encoded = self._tokenizer.encode_with_padding(
                inputs,
                return_tensors="pt",
                **kwargs,
            )
            return {k: v for k, v in encoded.items()}
        elif isinstance(inputs, dict):
            return inputs
        return {"text": inputs}

    def forward(self, model_inputs: Dict[str, Any], **kwargs) -> Any:
        """Run the model forward pass."""
        import torch

        device = next(self._model.parameters()).device
        for key in model_inputs:
            if hasattr(model_inputs[key], "to"):
                model_inputs[key] = model_inputs[key].to(device)

        with torch.no_grad():
            return self._model.generate(
                **model_inputs,
                max_new_tokens=self._generation_config.max_new_tokens,
                do_sample=self._generation_config.do_sample,
                temperature=self._generation_config.temperature,
                top_p=self._generation_config.top_p,
                top_k=self._generation_config.top_k,
                repetition_penalty=self._generation_config.repetition_penalty,
                use_cache=self._generation_config.use_cache,
                **kwargs,
            )

    def postprocess(self, model_outputs: Any, **kwargs) -> Any:
        """Postprocess model outputs into final format."""
        if self._config.custom_postprocessor:
            return self._config.custom_postprocessor(model_outputs, **kwargs)

        if hasattr(model_outputs, "sequences"):
            sequences = model_outputs.sequences
        else:
            sequences = model_outputs

        results = self._tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [{"generated_text": text} for text in results]

    def __call__(self, inputs: Any, **kwargs) -> Any:
        """Run the full pipeline: preprocess -> forward -> postprocess."""
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self.forward(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)


class TextGenerationPipeline(Pipeline):
    """Pipeline for causal language model text generation."""

    def __call__(self, text: str, **kwargs) -> List[Dict[str, str]]:
        """Generate text from a prompt.

        Returns a list of dicts with 'generated_text' key.
        """
        import torch

        input_ids = self._tokenizer.encode(text, return_tensors="pt")
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[-1]

        gen_kwargs = {
            "max_new_tokens": self._generation_config.max_new_tokens,
            "do_sample": self._generation_config.do_sample,
            "temperature": self._generation_config.temperature,
            "top_p": self._generation_config.top_p,
            "top_k": self._generation_config.top_k,
            "repetition_penalty": self._generation_config.repetition_penalty,
            "use_cache": self._generation_config.use_cache,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            output = self._model.generate(input_ids, **gen_kwargs)

        generated_ids = output[0, prompt_len:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return [{"generated_text": generated_text}]

    def generate_batch(self, texts: List[str], **kwargs) -> List[List[Dict[str, str]]]:
        """Generate text for a batch of prompts."""
        results = []
        for text in texts:
            results.append(self(text, **kwargs))
        return results


class Text2TextGenerationPipeline(Pipeline):
    """Pipeline for seq2seq (encoder-decoder) text generation."""

    def __call__(self, text: str, **kwargs) -> List[Dict[str, str]]:
        """Generate text using a text2text model."""
        import torch

        encoded = self._tokenizer.encode_with_padding([text], return_tensors="pt")
        device = next(self._model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        gen_kwargs = {
            "max_new_tokens": self._generation_config.max_new_tokens,
            "do_sample": self._generation_config.do_sample,
            "temperature": self._generation_config.temperature,
            "top_p": self._generation_config.top_p,
            "use_cache": self._generation_config.use_cache,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return [{"generated_text": generated_text}]


class ConversationalPipeline(Pipeline):
    """Pipeline for multi-turn conversational generation."""

    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Generate a response for a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        """
        import torch

        formatted = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if isinstance(formatted, str):
            input_ids = self._tokenizer.encode(formatted, return_tensors="pt")
        else:
            input_ids = torch.tensor([formatted])

        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[-1]

        gen_kwargs = {
            "max_new_tokens": self._generation_config.max_new_tokens,
            "do_sample": self._generation_config.do_sample,
            "temperature": self._generation_config.temperature,
            "top_p": self._generation_config.top_p,
            "use_cache": self._generation_config.use_cache,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            output = self._model.generate(input_ids, **gen_kwargs)

        generated_ids = output[0, prompt_len:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return [{"generated_text": generated_text}]


class CustomPipeline(Pipeline):
    """Pipeline with custom preprocessor, forward, and postprocessor functions."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: PipelineConfig,
        preprocess_fn: Optional[Callable] = None,
        forward_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        super().__init__(model, tokenizer, config)
        self._custom_preprocess = preprocess_fn
        self._custom_forward = forward_fn
        self._custom_postprocess = postprocess_fn

    def preprocess(self, inputs: Any, **kwargs) -> Any:
        if self._custom_preprocess:
            return self._custom_preprocess(inputs, **kwargs)
        return super().preprocess(inputs, **kwargs)

    def forward(self, model_inputs: Any, **kwargs) -> Any:
        if self._custom_forward:
            return self._custom_forward(self._model, model_inputs, **kwargs)
        return super().forward(model_inputs, **kwargs)

    def postprocess(self, model_outputs: Any, **kwargs) -> Any:
        if self._custom_postprocess:
            return self._custom_postprocess(model_outputs, **kwargs)
        return super().postprocess(model_outputs, **kwargs)


class PipelineFactory:
    """Factory for creating pipelines from configurations."""

    @staticmethod
    def create(
        model: Any,
        tokenizer: Any,
        pipeline_type: PipelineType = PipelineType.TEXT_GENERATION,
        config: Optional[PipelineConfig] = None,
        **kwargs,
    ) -> Pipeline:
        """Create a pipeline of the specified type."""
        if config is None:
            config = PipelineConfig(pipeline_type=pipeline_type, **kwargs)
        else:
            config.pipeline_type = pipeline_type

        pipeline_map = {
            PipelineType.TEXT_GENERATION: TextGenerationPipeline,
            PipelineType.TEXT2TEXT_GENERATION: Text2TextGenerationPipeline,
            PipelineType.CONVERSATIONAL: ConversationalPipeline,
            PipelineType.CUSTOM: CustomPipeline,
        }

        pipeline_cls = pipeline_map.get(pipeline_type, TextGenerationPipeline)

        if pipeline_type == PipelineType.CUSTOM:
            return pipeline_cls(
                model=model,
                tokenizer=tokenizer,
                config=config,
                preprocess_fn=kwargs.get("preprocess_fn"),
                forward_fn=kwargs.get("forward_fn"),
                postprocess_fn=kwargs.get("postprocess_fn"),
            )

        return pipeline_cls(model=model, tokenizer=tokenizer, config=config)

    @staticmethod
    def from_pretrained(
        model_path: str,
        pipeline_type: PipelineType = PipelineType.TEXT_GENERATION,
        device: str = "auto",
        torch_dtype: str = "float16",
        trust_remote_code: bool = False,
        quantization_config: Optional[Any] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Pipeline:
        """Create a pipeline by loading a model from a pretrained path."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

        torch_dtype_val = getattr(torch, torch_dtype, torch.float16)
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch_dtype_val,
            "trust_remote_code": trust_remote_code,
            "device_map": device,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        if pipeline_type == PipelineType.TEXT2TEXT_GENERATION:
            model = AutoModelForSeq2SeqLM.from_pretrained(**load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = PipelineConfig(
            pipeline_type=pipeline_type,
            model_id=model_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            generation_config=generation_config,
        )

        return PipelineFactory.create(model, tokenizer, pipeline_type, config)
