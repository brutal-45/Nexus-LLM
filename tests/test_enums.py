"""Tests for enumerations (ModelType, DeviceType, PrecisionType, TaskType, ChatRole)."""
import pytest

from nexus_llm.enums import (
    ModelType,
    DeviceType,
    PrecisionType,
    TaskType,
    ChatRole,
    MessageType,
    TrainingStage,
)


class TestModelType:
    """Test ModelType enumeration."""

    def test_all_model_types_exist(self):
        expected = ["causal_lm", "seq2seq_lm", "masked_lm", "instruction",
                     "chat", "code", "embedding", "rlhf", "dpo", "multimodal"]
        for val in expected:
            assert ModelType(val) is not None

    def test_str_representation(self):
        assert str(ModelType.CAUSAL_LM) == "causal_lm"
        assert str(ModelType.CHAT) == "chat"

    def test_description_property(self):
        assert "Causal" in ModelType.CAUSAL_LM.description
        assert "Chat" in ModelType.CHAT.description
        assert "Code" in ModelType.CODE.description
        assert "Embedding" in ModelType.EMBEDDING.description

    def test_is_string_enum(self):
        assert ModelType.CAUSAL_LM.value == "causal_lm"
        assert isinstance(ModelType.CAUSAL_LM, str)

    def test_from_value(self):
        m = ModelType("chat")
        assert m == ModelType.CHAT


class TestDeviceType:
    """Test DeviceType enumeration."""

    def test_all_device_types(self):
        expected = ["auto", "cpu", "cuda", "mps", "tpu", "xpu"]
        for val in expected:
            assert DeviceType(val) is not None

    def test_str_representation(self):
        assert str(DeviceType.CUDA) == "cuda"
        assert str(DeviceType.AUTO) == "auto"

    def test_description_property(self):
        assert "Automatic" in DeviceType.AUTO.description
        assert "NVIDIA" in DeviceType.CUDA.description
        assert "Apple" in DeviceType.MPS.description

    def test_detect_returns_device_type(self):
        result = DeviceType.detect()
        assert isinstance(result, DeviceType)
        assert result in (DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS)


class TestPrecisionType:
    """Test PrecisionType enumeration."""

    def test_all_precision_types(self):
        expected = ["fp32", "fp16", "bf16", "int8", "int4",
                     "gptq_int4", "gptq_int8", "awq_int4", "mixed"]
        for val in expected:
            assert PrecisionType(val) is not None

    def test_bits_property(self):
        assert PrecisionType.FP32.bits == 32
        assert PrecisionType.FP16.bits == 16
        assert PrecisionType.BF16.bits == 16
        assert PrecisionType.INT8.bits == 8
        assert PrecisionType.INT4.bits == 4
        assert PrecisionType.GPTQ_INT4.bits == 4
        assert PrecisionType.AWQ_INT4.bits == 4

    def test_is_quantized_property(self):
        assert PrecisionType.FP32.is_quantized is False
        assert PrecisionType.FP16.is_quantized is False
        assert PrecisionType.INT8.is_quantized is True
        assert PrecisionType.INT4.is_quantized is True
        assert PrecisionType.GPTQ_INT4.is_quantized is True
        assert PrecisionType.AWQ_INT4.is_quantized is True
        assert PrecisionType.MIXED.is_quantized is False


class TestTaskType:
    """Test TaskType enumeration."""

    def test_all_task_types(self):
        expected = ["text_generation", "text_classification", "token_classification",
                     "question_answering", "summarization", "translation",
                     "code_generation", "embedding", "chat",
                     "instruction_following", "fine_tuning", "evaluation", "benchmarking"]
        for val in expected:
            assert TaskType(val) is not None

    def test_str_representation(self):
        assert str(TaskType.TEXT_GENERATION) == "text_generation"
        assert str(TaskType.CHAT) == "chat"


class TestChatRole:
    """Test ChatRole enumeration."""

    def test_all_chat_roles(self):
        expected = ["system", "user", "assistant", "function", "tool"]
        for val in expected:
            assert ChatRole(val) is not None

    def test_str_representation(self):
        assert str(ChatRole.SYSTEM) == "system"
        assert str(ChatRole.USER) == "user"
        assert str(ChatRole.ASSISTANT) == "assistant"


class TestTrainingStage:
    """Test TrainingStage enumeration."""

    def test_is_active_property(self):
        assert TrainingStage.TRAINING.is_active is True
        assert TrainingStage.EVALUATING.is_active is True
        assert TrainingStage.COMPLETED.is_active is False
        assert TrainingStage.FAILED.is_active is False

    def test_is_terminal_property(self):
        assert TrainingStage.COMPLETED.is_terminal is True
        assert TrainingStage.FAILED.is_terminal is True
        assert TrainingStage.INTERRUPTED.is_terminal is True
        assert TrainingStage.TRAINING.is_terminal is False

    def test_all_stages_exist(self):
        expected = ["initializing", "preparing_data", "loading_model", "configuring",
                     "training", "evaluating", "saving_checkpoint", "completed",
                     "failed", "interrupted"]
        for val in expected:
            assert TrainingStage(val) is not None
