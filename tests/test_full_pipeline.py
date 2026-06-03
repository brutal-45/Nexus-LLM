"""Tests for full pipeline end-to-end."""
import pytest

from nexus_llm.enums import ChatRole, ModelType, DeviceType
from nexus_llm.types import (
    Message,
    Conversation,
    GenerationConfig,
    ModelInfo,
    ChatConfig,
    ServerConfig,
    TrainingConfig,
)
from nexus_llm.events import EventBus, Event
from nexus_llm.registry import Registry
from nexus_llm.state import StateManager
from nexus_llm.config_loader import ConfigLoader


class TestMessageToConversationPipeline:
    """Test message creation through conversation building."""

    def test_full_conversation_flow(self):
        # Create a conversation
        conv = Conversation(model="test-model")
        conv.add_message(ChatRole.SYSTEM, "You are helpful.")
        conv.add_message(ChatRole.USER, "What is AI?")
        conv.add_message(ChatRole.ASSISTANT, "AI is artificial intelligence.")

        assert len(conv.messages) == 3
        assert conv.messages[0].role == ChatRole.SYSTEM
        assert conv.messages[1].role == ChatRole.USER
        assert conv.messages[2].role == ChatRole.ASSISTANT

        # Serialize and deserialize
        d = conv.to_dict()
        conv2 = Conversation.from_dict(d)
        assert len(conv2.messages) == 3
        assert conv2.model == "test-model"


class TestEventBusWithRegistryPipeline:
    """Test EventBus + Registry integration."""

    def test_register_model_and_emit_event(self):
        bus = EventBus()
        registry = Registry(name="models", allow_overwrite=True)

        events_received = []
        bus.subscribe(event_type="model.registered",
                      handler=lambda e: events_received.append(e))

        # Register a model
        model_info = ModelInfo(name="gpt2", model_type=ModelType.CAUSAL_LM)
        registry.register("gpt2", model_info)

        # Emit event
        bus.publish(Event(event_type="model.registered",
                          data={"model_name": "gpt2"}, source="test"))

        assert len(events_received) == 1
        assert registry.get("gpt2").name == "gpt2"


class TestConfigToStatePipeline:
    """Test config loading into state management."""

    def test_config_values_applied_to_state(self):
        loader = ConfigLoader()
        config = loader.load()
        state = StateManager()

        # Apply config to state
        if "default_model" in config:
            state.set("model.current", config["default_model"])
        if "port" in config:
            state.set("server.port", config["port"])

        assert state.get("model.current") is not None
        assert state.get("server.port") is not None


class TestChatConfigToConversationPipeline:
    """Test ChatConfig to Conversation flow."""

    def test_chat_config_creates_conversation(self):
        chat_config = ChatConfig(
            model="gpt2",
            system_prompt="You are a coding assistant.",
            temperature=0.5,
            max_tokens=1024,
        )

        conv = Conversation(model=chat_config.model)
        if chat_config.system_prompt:
            conv.add_message(ChatRole.SYSTEM, chat_config.system_prompt)

        assert conv.model == "gpt2"
        assert len(conv.messages) == 1
        assert conv.messages[0].role == ChatRole.SYSTEM


class TestGenerationConfigPipeline:
    """Test GenerationConfig through the pipeline."""

    def test_config_to_dict_and_back(self):
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
        )
        d = config.to_dict()
        assert d["max_tokens"] == 512
        assert d["temperature"] == 0.3

        # Simulate reconstruction
        config2 = GenerationConfig(**d)
        assert config2.max_tokens == 512
        assert config2.temperature == 0.3
