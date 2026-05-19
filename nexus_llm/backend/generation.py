"""Generation configuration for Nexus-LLM backend.

Provides GenerationConfig dataclass and preset configurations for common
generation scenarios (creative, precise, balanced, code generation).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


@dataclass
class GenerationConfig:
    """Complete generation configuration with all supported parameters.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        max_length: Maximum total length of input + output.
        temperature: Sampling temperature. Higher = more random.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        repetition_penalty: Penalty for repeating tokens.
        no_repeat_ngram_size: Prevent repeating n-grams of this size.
        num_beams: Number of beams for beam search.
        num_return_sequences: Number of sequences to return.
        length_penalty: Exponential penalty for length in beam search.
        early_stopping: Whether to stop beam search early.
        do_sample: Whether to use sampling (True) or greedy (False).
        num_beam_groups: Number of groups for diverse beam search.
        diversity_penalty: Penalty for diversity in diverse beam search.
        min_new_tokens: Minimum number of new tokens to generate.
        min_length: Minimum total length of output.
        encoder_no_repeat_ngram_size: No-repeat n-gram size for encoder.
        bad_words_ids: Token IDs to suppress in generation.
        force_words_ids: Token IDs to force in generation.
        renormalize_logits: Whether to renormalize logits after processing.
        remove_invalid_values: Remove NaN/Inf from logits.
        output_scores: Whether to return prediction scores.
        output_logits: Whether to return raw logits.
        return_dict_in_generate: Return dict instead of tuple.
        pad_token_id: Padding token ID.
        bos_token_id: Beginning-of-sequence token ID.
        eos_token_id: End-of-sequence token ID(s).
        use_cache: Whether to use KV cache.
        typical_p: Typical sampling threshold.
        epsilon_cutoff: Epsilon sampling cutoff.
        eta_cutoff: Eta sampling cutoff.
        guidance_scale: Classifier-free guidance scale.
        prefix_allowed_tokens_fn: Function to constrain token generation.
        seed: Random seed for reproducibility.
    """

    max_new_tokens: int = 512
    max_length: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    num_return_sequences: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    do_sample: bool = True
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    min_new_tokens: int = 0
    min_length: int = 0
    encoder_no_repeat_ngram_size: int = 0
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    renormalize_logits: bool = False
    remove_invalid_values: bool = False
    output_scores: bool = False
    output_logits: bool = False
    return_dict_in_generate: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Any] = None
    use_cache: bool = True
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    guidance_scale: Optional[float] = None
    prefix_allowed_tokens_fn: Optional[Any] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create a GenerationConfig from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationConfig":
        """Create a GenerationConfig from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def merge(self, other: "GenerationConfig") -> "GenerationConfig":
        """Merge another config into this one. Non-default values from other take precedence."""
        merged = asdict(self)
        defaults = GenerationConfig()
        for k, v in asdict(other).items():
            default_val = getattr(defaults, k)
            if v != default_val:
                merged[k] = v
        return GenerationConfig(**merged)

    def validate(self) -> List[str]:
        """Validate the configuration and return a list of warnings."""
        warnings = []
        if self.temperature <= 0:
            warnings.append(f"Temperature must be positive, got {self.temperature}")
        if not 0 < self.top_p <= 1.0:
            warnings.append(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            warnings.append(f"top_k must be non-negative, got {self.top_k}")
        if self.repetition_penalty <= 0:
            warnings.append(f"repetition_penalty must be positive, got {self.repetition_penalty}")
        if self.num_beams < 1:
            warnings.append(f"num_beams must be >= 1, got {self.num_beams}")
        if self.num_return_sequences > self.num_beams:
            warnings.append(
                f"num_return_sequences ({self.num_return_sequences}) cannot exceed "
                f"num_beams ({self.num_beams})"
            )
        if self.max_new_tokens < 1:
            warnings.append(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.max_length is not None and self.max_length < 1:
            warnings.append(f"max_length must be >= 1, got {self.max_length}")
        if self.num_beam_groups > 1 and self.num_beams % self.num_beam_groups != 0:
            warnings.append(
                f"num_beams ({self.num_beams}) must be divisible by "
                f"num_beam_groups ({self.num_beam_groups})"
            )
        return warnings


class GenerationPresets:
    """Preset generation configurations for common use cases."""

    @staticmethod
    def creative() -> GenerationConfig:
        """Creative writing preset: high temperature, top-p sampling."""
        return GenerationConfig(
            max_new_tokens=1024,
            temperature=1.2,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            do_sample=True,
            typical_p=1.0,
            use_cache=True,
        )

    @staticmethod
    def precise() -> GenerationConfig:
        """Precise/factual preset: low temperature, minimal randomness."""
        return GenerationConfig(
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            top_k=10,
            repetition_penalty=1.05,
            do_sample=True,
            use_cache=True,
        )

    @staticmethod
    def balanced() -> GenerationConfig:
        """Balanced preset: moderate temperature and sampling."""
        return GenerationConfig(
            max_new_tokens=768,
            temperature=0.7,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.08,
            do_sample=True,
            use_cache=True,
        )

    @staticmethod
    def code() -> GenerationConfig:
        """Code generation preset: low temperature, deterministic."""
        return GenerationConfig(
            max_new_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.1,
            do_sample=True,
            use_cache=True,
            remove_invalid_values=True,
        )

    @staticmethod
    def greedy() -> GenerationConfig:
        """Greedy decoding preset: deterministic, no sampling."""
        return GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    @staticmethod
    def beam_search() -> GenerationConfig:
        """Beam search preset: 4 beams, length penalty."""
        return GenerationConfig(
            max_new_tokens=512,
            num_beams=4,
            num_return_sequences=1,
            length_penalty=1.2,
            early_stopping=True,
            do_sample=False,
            use_cache=True,
        )

    @staticmethod
    def diverse() -> GenerationConfig:
        """Diverse beam search preset: multiple beam groups."""
        return GenerationConfig(
            max_new_tokens=512,
            num_beams=6,
            num_beam_groups=3,
            diversity_penalty=2.0,
            num_return_sequences=3,
            do_sample=False,
            use_cache=True,
        )

    @staticmethod
    def chat() -> GenerationConfig:
        """Chat/conversational preset."""
        return GenerationConfig(
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            use_cache=True,
        )

    @staticmethod
    def summarize() -> GenerationConfig:
        """Summarization preset: shorter output, lower temperature."""
        return GenerationConfig(
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.9,
            top_k=30,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
            do_sample=True,
            use_cache=True,
        )

    @staticmethod
    def get_preset(name: str) -> GenerationConfig:
        """Get a preset by name. Returns balanced if name not found."""
        presets = {
            "creative": GenerationPresets.creative,
            "precise": GenerationPresets.precise,
            "balanced": GenerationPresets.balanced,
            "code": GenerationPresets.code,
            "greedy": GenerationPresets.greedy,
            "beam_search": GenerationPresets.beam_search,
            "diverse": GenerationPresets.diverse,
            "chat": GenerationPresets.chat,
            "summarize": GenerationPresets.summarize,
        }
        factory = presets.get(name.lower(), GenerationPresets.balanced)
        return factory()

    @staticmethod
    def list_presets() -> List[str]:
        """List all available preset names."""
        return [
            "creative", "precise", "balanced", "code", "greedy",
            "beam_search", "diverse", "chat", "summarize",
        ]
