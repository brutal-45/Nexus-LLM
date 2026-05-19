"""
Nexus-LLM Configuration Validators

Provides validation for configuration values including type checking,
range validation, dependency validation, and cross-field consistency checks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


class ValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, field: str = "") -> None:
        self.field = field
        super().__init__(f"{field}: {message}" if field else message)


@dataclass
class ValidationRule:
    """A single validation rule for a configuration field."""
    field: str
    rule_type: str  # type, range, choice, regex, dependency, custom
    params: dict[str, Any] | None = None
    message: str = ""
    condition: Any | None = None


# Validation rules for all configuration sections
VALIDATION_RULES: list[ValidationRule] = [
    # Model
    ValidationRule("model.name", "type", {"expected": "str"}, "Model name must be a string"),
    ValidationRule("model.device", "choice", {"options": ["auto", "cpu", "cuda", "mps"]}, "Device must be auto, cpu, cuda, or mps"),
    ValidationRule("model.dtype", "choice", {"options": ["float32", "float16", "bfloat16"]}, "dtype must be float32, float16, or bfloat16"),

    # Generation
    ValidationRule("generation.temperature", "type", {"expected": "num"}, "Temperature must be a number"),
    ValidationRule("generation.temperature", "range", {"min": 0.0, "max": 2.0}, "Temperature must be between 0.0 and 2.0"),
    ValidationRule("generation.top_p", "type", {"expected": "num"}, "Top-p must be a number"),
    ValidationRule("generation.top_p", "range", {"min": 0.0, "max": 1.0}, "Top-p must be between 0.0 and 1.0"),
    ValidationRule("generation.top_k", "type", {"expected": "int"}, "Top-k must be an integer"),
    ValidationRule("generation.top_k", "range", {"min": 1, "max": 1000}, "Top-k must be between 1 and 1000"),
    ValidationRule("generation.max_tokens", "type", {"expected": "int"}, "Max tokens must be an integer"),
    ValidationRule("generation.max_tokens", "range", {"min": 1, "max": 32768}, "Max tokens must be between 1 and 32768"),
    ValidationRule("generation.min_tokens", "type", {"expected": "int"}, "Min tokens must be an integer"),
    ValidationRule("generation.min_tokens", "range", {"min": 0, "max": 32768}, "Min tokens must be between 0 and 32768"),
    ValidationRule("generation.repetition_penalty", "type", {"expected": "num"}, "Repetition penalty must be a number"),
    ValidationRule("generation.repetition_penalty", "range", {"min": 1.0, "max": 2.0}, "Repetition penalty must be between 1.0 and 2.0"),
    ValidationRule("generation.frequency_penalty", "type", {"expected": "num"}, "Frequency penalty must be a number"),
    ValidationRule("generation.frequency_penalty", "range", {"min": -2.0, "max": 2.0}, "Frequency penalty must be between -2.0 and 2.0"),
    ValidationRule("generation.presence_penalty", "type", {"expected": "num"}, "Presence penalty must be a number"),
    ValidationRule("generation.presence_penalty", "range", {"min": -2.0, "max": 2.0}, "Presence penalty must be between -2.0 and 2.0"),
    ValidationRule("generation.num_beams", "type", {"expected": "int"}, "Num beams must be an integer"),
    ValidationRule("generation.num_beams", "range", {"min": 1, "max": 32}, "Num beams must be between 1 and 32"),

    # Quantization
    ValidationRule("quantization.bits", "choice", {"options": [4, 8]}, "Quantization bits must be 4 or 8"),
    ValidationRule("quantization.group_size", "type", {"expected": "int"}, "Group size must be an integer"),
    ValidationRule("quantization.group_size", "range", {"min": 32, "max": 512}, "Group size must be between 32 and 512"),

    # Training
    ValidationRule("training.num_train_epochs", "type", {"expected": "num"}, "Num epochs must be a number"),
    ValidationRule("training.num_train_epochs", "range", {"min": 1, "max": 1000}, "Num epochs must be between 1 and 1000"),
    ValidationRule("training.per_device_train_batch_size", "type", {"expected": "int"}, "Batch size must be an integer"),
    ValidationRule("training.per_device_train_batch_size", "range", {"min": 1, "max": 256}, "Batch size must be between 1 and 256"),
    ValidationRule("training.learning_rate", "type", {"expected": "num"}, "Learning rate must be a number"),
    ValidationRule("training.learning_rate", "range", {"min": 1e-8, "max": 1.0}, "Learning rate must be between 1e-8 and 1.0"),
    ValidationRule("training.weight_decay", "type", {"expected": "num"}, "Weight decay must be a number"),
    ValidationRule("training.weight_decay", "range", {"min": 0.0, "max": 1.0}, "Weight decay must be between 0.0 and 1.0"),
    ValidationRule("training.max_grad_norm", "type", {"expected": "num"}, "Max grad norm must be a number"),
    ValidationRule("training.max_grad_norm", "range", {"min": 0.0, "max": 100.0}, "Max grad norm must be between 0.0 and 100.0"),
    ValidationRule("training.lr_scheduler_type", "choice", {"options": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]}, "Invalid LR scheduler type"),
    ValidationRule("training.optimizer", "choice", {"options": ["adamw_hf", "adamw_torch", "adamw_apex_fused", "adafactor"]}, "Invalid optimizer"),

    # LoRA
    ValidationRule("lora.r", "type", {"expected": "int"}, "LoRA r must be an integer"),
    ValidationRule("lora.r", "range", {"min": 1, "max": 512}, "LoRA r must be between 1 and 512"),
    ValidationRule("lora.lora_alpha", "type", {"expected": "int"}, "LoRA alpha must be an integer"),
    ValidationRule("lora.lora_alpha", "range", {"min": 1, "max": 1024}, "LoRA alpha must be between 1 and 1024"),
    ValidationRule("lora.lora_dropout", "type", {"expected": "num"}, "LoRA dropout must be a number"),
    ValidationRule("lora.lora_dropout", "range", {"min": 0.0, "max": 1.0}, "LoRA dropout must be between 0.0 and 1.0"),
    ValidationRule("lora.bias", "choice", {"options": ["none", "all", "lora_only"]}, "LoRA bias must be none, all, or lora_only"),
    ValidationRule("lora.task_type", "choice", {"options": ["CAUSAL_LM", "SEQ_2_SEQ_LM", "SEQ_CLS", "TOKEN_CLS", "QUESTION_ANS"]}, "Invalid LoRA task type"),

    # Server
    ValidationRule("server.port", "type", {"expected": "int"}, "Port must be an integer"),
    ValidationRule("server.port", "range", {"min": 1, "max": 65535}, "Port must be between 1 and 65535"),
    ValidationRule("server.workers", "type", {"expected": "int"}, "Workers must be an integer"),
    ValidationRule("server.workers", "range", {"min": 1, "max": 64}, "Workers must be between 1 and 64"),
    ValidationRule("server.max_request_size", "type", {"expected": "int"}, "Max request size must be an integer"),
    ValidationRule("server.max_request_size", "range", {"min": 1024, "max": 1073741824}, "Max request size must be between 1KB and 1GB"),

    # Rate limiting
    ValidationRule("rate_limiting.requests_per_minute", "type", {"expected": "int"}, "Requests per minute must be an integer"),
    ValidationRule("rate_limiting.requests_per_minute", "range", {"min": 1, "max": 100000}, "Requests per minute must be between 1 and 100000"),

    # UI
    ValidationRule("ui.theme", "choice", {"options": ["dark", "light", "monokai", "solarized", "dracula", "nord"]}, "Invalid theme name"),
    ValidationRule("ui.input_mode", "choice", {"options": ["emacs", "vi"]}, "Input mode must be emacs or vi"),
    ValidationRule("ui.panel_style", "choice", {"options": ["rounded", "heavy", "double", "minimal", "ascii"]}, "Invalid panel style"),
]


class ConfigValidator:
    """Validates configuration values against defined rules.

    Supports type checking, range validation, choice validation,
    regex pattern matching, dependency checking, and custom validators.
    """

    def __init__(self, rules: list[ValidationRule] | None = None) -> None:
        self._rules = rules or VALIDATION_RULES
        self._custom_validators: dict[str, Any] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule.

        Args:
            rule: The ValidationRule to add.
        """
        self._rules.append(rule)

    def add_custom_validator(self, field: str, validator: Any) -> None:
        """Add a custom validator function for a field.

        Args:
            field: Dot-notation field path.
            validator: Callable that takes a value and returns an error string or None.
        """
        self._custom_validators[field] = validator

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validate a configuration dictionary against all rules.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            List of error message strings (empty if valid).
        """
        errors: list[str] = []

        for rule in self._rules:
            value = self._get_value(config, rule.field)

            # Skip validation if the field doesn't exist and isn't required
            if value is None:
                continue

            error = self._validate_rule(rule, value, config)
            if error:
                errors.append(error)

        # Run custom validators
        for field_path, validator in self._custom_validators.items():
            value = self._get_value(config, field_path)
            if value is not None:
                try:
                    error = validator(value)
                    if error:
                        errors.append(f"{field_path}: {error}")
                except Exception as exc:
                    errors.append(f"{field_path}: validation error: {exc}")

        # Run dependency checks
        dep_errors = self._check_dependencies(config)
        errors.extend(dep_errors)

        return errors

    def validate_field(self, config: dict[str, Any], field: str) -> list[str]:
        """Validate a single configuration field.

        Args:
            config: Configuration dictionary.
            field: Dot-notation field path.

        Returns:
            List of error messages for the field.
        """
        value = self._get_value(config, field)
        if value is None:
            return [f"{field}: field not found"]

        errors = []
        for rule in self._rules:
            if rule.field == field:
                error = self._validate_rule(rule, value, config)
                if error:
                    errors.append(error)
        return errors

    def validate_or_raise(self, config: dict[str, Any]) -> None:
        """Validate configuration and raise on errors.

        Args:
            config: Configuration dictionary.

        Raises:
            ValidationError: If any validation errors are found.
        """
        errors = self.validate(config)
        if errors:
            raise ValidationError("; ".join(errors))

    def _validate_rule(self, rule: ValidationRule, value: Any, config: dict[str, Any]) -> str | None:
        """Validate a single rule against a value.

        Args:
            rule: The validation rule.
            value: The value to validate.
            config: Full config for dependency checks.

        Returns:
            Error message string, or None if valid.
        """
        if rule.rule_type == "type":
            return self._validate_type(rule, value)
        elif rule.rule_type == "range":
            return self._validate_range(rule, value)
        elif rule.rule_type == "choice":
            return self._validate_choice(rule, value)
        elif rule.rule_type == "regex":
            return self._validate_regex(rule, value)
        elif rule.rule_type == "dependency":
            return self._validate_dependency(rule, value, config)
        return None

    def _validate_type(self, rule: ValidationRule, value: Any) -> str | None:
        """Validate the type of a value."""
        expected = rule.params.get("expected", "") if rule.params else ""

        if expected == "str":
            if not isinstance(value, str):
                return rule.message or f"Expected string, got {type(value).__name__}"
        elif expected == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return rule.message or f"Expected integer, got {type(value).__name__}"
        elif expected == "num":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return rule.message or f"Expected number, got {type(value).__name__}"
        elif expected == "bool":
            if not isinstance(value, bool):
                return rule.message or f"Expected boolean, got {type(value).__name__}"
        elif expected == "list":
            if not isinstance(value, list):
                return rule.message or f"Expected list, got {type(value).__name__}"
        elif expected == "dict":
            if not isinstance(value, dict):
                return rule.message or f"Expected dict, got {type(value).__name__}"

        return None

    def _validate_range(self, rule: ValidationRule, value: Any) -> str | None:
        """Validate that a value falls within a range."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return rule.message or "Value must be a number for range validation"

        params = rule.params or {}
        min_val = params.get("min")
        max_val = params.get("max")

        if min_val is not None and value < min_val:
            return rule.message or f"Value {value} is below minimum {min_val}"
        if max_val is not None and value > max_val:
            return rule.message or f"Value {value} exceeds maximum {max_val}"

        return None

    def _validate_choice(self, rule: ValidationRule, value: Any) -> str | None:
        """Validate that a value is one of the allowed choices."""
        params = rule.params or {}
        options = params.get("options", [])

        if value not in options:
            return rule.message or f"Value '{value}' not in allowed options: {options}"

        return None

    def _validate_regex(self, rule: ValidationRule, value: Any) -> str | None:
        """Validate that a value matches a regex pattern."""
        if not isinstance(value, str):
            return rule.message or "Value must be a string for regex validation"

        params = rule.params or {}
        pattern = params.get("pattern", "")

        if not re.match(pattern, value):
            return rule.message or f"Value '{value}' does not match pattern '{pattern}'"

        return None

    def _validate_dependency(self, rule: ValidationRule, value: Any, config: dict[str, Any]) -> str | None:
        """Validate a dependency between fields."""
        params = rule.params or {}
        depends_on = params.get("depends_on", "")
        depends_value = params.get("depends_value")

        dep_value = self._get_value(config, depends_on)
        if depends_value is not None and dep_value != depends_value:
            return None  # Dependency not active, skip validation

        # If dependency is met, validate the current field
        return None

    def _check_dependencies(self, config: dict[str, Any]) -> list[str]:
        """Check cross-field dependencies and consistency.

        Args:
            config: Full configuration dictionary.

        Returns:
            List of dependency error messages.
        """
        errors: list[str] = []

        # min_tokens must be <= max_tokens
        min_tokens = self._get_value(config, "generation.min_tokens")
        max_tokens = self._get_value(config, "generation.max_tokens")
        if (min_tokens is not None and max_tokens is not None
                and isinstance(min_tokens, (int, float))
                and isinstance(max_tokens, (int, float))
                and min_tokens > max_tokens):
            errors.append("generation.min_tokens cannot exceed generation.max_tokens")

        # LoRA target modules required when LoRA is enabled
        lora_enabled = self._get_value(config, "lora.enabled")
        lora_targets = self._get_value(config, "lora.target_modules")
        if lora_enabled and (not lora_targets or len(lora_targets) == 0):
            errors.append("lora.target_modules must be specified when lora is enabled")

        # Auth secret key must be changed from default when auth is enabled
        auth_enabled = self._get_value(config, "auth.enabled")
        secret_key = self._get_value(config, "auth.secret_key")
        if auth_enabled and secret_key == "change-me-in-production":
            errors.append("auth.secret_key must be changed from default when auth is enabled")

        # BF16 requires CUDA
        bf16 = self._get_value(config, "training.bf16")
        device = self._get_value(config, "model.device")
        if bf16 and device == "cpu":
            errors.append("training.bf16 is not supported on CPU device")

        # Quantization requires model path or name
        quant_enabled = self._get_value(config, "quantization.enabled")
        model_name = self._get_value(config, "model.name")
        model_path = self._get_value(config, "model.path")
        if quant_enabled and not model_name and not model_path:
            errors.append("model.name or model.path must be specified when quantization is enabled")

        # Eval steps must be positive when evaluation strategy is 'steps'
        eval_strategy = self._get_value(config, "training.evaluation_strategy")
        eval_steps = self._get_value(config, "training.eval_steps")
        if eval_strategy == "steps" and (not eval_steps or eval_steps <= 0):
            errors.append("training.eval_steps must be positive when evaluation_strategy is 'steps'")

        # Save steps must be positive when save strategy is 'steps'
        save_strategy = self._get_value(config, "training.save_strategy")
        save_steps = self._get_value(config, "training.save_steps")
        if save_strategy == "steps" and (not save_steps or save_steps <= 0):
            errors.append("training.save_steps must be positive when save_strategy is 'steps'")

        # Grad accumulation steps must be positive
        grad_acc = self._get_value(config, "training.gradient_accumulation_steps")
        if grad_acc is not None and isinstance(grad_acc, int) and grad_acc < 1:
            errors.append("training.gradient_accumulation_steps must be at least 1")

        return errors

    @staticmethod
    def _get_value(config: dict[str, Any], key: str) -> Any:
        """Get a value from a nested dictionary by dot-notation key.

        Args:
            config: Nested configuration dictionary.
            key: Dot-separated key path.

        Returns:
            The value at the key path, or None if not found.
        """
        parts = key.split(".")
        current = config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
