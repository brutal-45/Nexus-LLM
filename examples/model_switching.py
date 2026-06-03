#!/usr/bin/env python3
"""
Model Switching Example - Nexus-LLM
=====================================
Demonstrates how to dynamically switch between different models
during a conversation or across tasks.
"""

from nexus_llm import InferenceEngine, Conversation, ModelRegistry


def main():
    # Register available models
    registry = ModelRegistry()
    registry.register(
        name="fast-model",
        model_name="nexus-3b-chat",
        device="auto",
        description="Fast model for quick responses",
    )
    registry.register(
        name="quality-model",
        model_name="nexus-13b-chat",
        device="auto",
        description="Larger model for high-quality responses",
    )
    registry.register(
        name="code-model",
        model_name="nexus-7b-code",
        device="auto",
        description="Specialized model for code generation",
    )

    # List available models
    print("Available models:")
    for model in registry.list_models():
        print(f"  - {model.name}: {model.description}")
    print()

    conversation = Conversation(
        system_prompt="You are a versatile AI assistant capable of both general chat and coding tasks."
    )

    # Use the fast model for a simple question
    engine = registry.get("fast-model")
    conversation.add_user_message("What is 2 + 2?")
    response = engine.chat(conversation)
    print(f"[fast-model] User: What is 2 + 2?")
    print(f"[fast-model] Assistant: {response.text}\n")

    # Switch to the code model for a coding question
    engine = registry.get("code-model")
    conversation.add_user_message("Write a Python function to compute Fibonacci numbers.")
    response = engine.chat(conversation)
    print(f"[code-model] User: Write a Python function to compute Fibonacci numbers.")
    print(f"[code-model] Assistant: {response.text}\n")

    # Switch to the quality model for a nuanced answer
    engine = registry.get("quality-model")
    conversation.add_user_message(
        "Explain the philosophical implications of artificial general intelligence."
    )
    response = engine.chat(conversation)
    print(f"[quality-model] User: Explain the philosophical implications of AGI.")
    print(f"[quality-model] Assistant: {response.text}\n")

    # Auto-switch based on task type
    print("--- Auto-switching based on task classification ---")
    router = registry.create_router(strategy="task-classifier")
    tasks = [
        "Summarize this article in one sentence.",
        "Debug this Python code: def add(a, b) return a + b",
        "Write an essay on climate change policy.",
    ]
    for task in tasks:
        selected = router.route(task)
        print(f"Task: {task}")
        print(f"  -> Routed to: {selected.name} ({selected.description})")


if __name__ == "__main__":
    main()
