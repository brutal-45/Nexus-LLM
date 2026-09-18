#!/usr/bin/env python3
"""
Basic Chat Example - Nexus-LLM
================================
Demonstrates how to use the InferenceEngine for a simple chat interaction.
"""

from nexus_llm import InferenceEngine, Conversation

def main():
    # Initialize the inference engine with a model
    engine = InferenceEngine(
        model_name="nexus-7b-chat",
        device="auto",           # Automatically select GPU/CPU
        dtype="auto",            # Automatically select precision
        max_memory=None,         # Use default memory mapping
    )

    # Create a new conversation with system prompt
    conversation = Conversation(
        system_prompt="You are a helpful, harmless, and honest AI assistant."
    )

    # First message
    conversation.add_user_message("What is the capital of France?")
    response = engine.chat(conversation)
    print(f"User: What is the capital of France?")
    print(f"Assistant: {response.text}\n")

    # Follow-up message (context is preserved)
    conversation.add_user_message("What about its population?")
    response = engine.chat(conversation)
    print(f"User: What about its population?")
    print(f"Assistant: {response.text}\n")

    # Print conversation metadata
    print(f"Total tokens used: {conversation.total_tokens}")
    print(f"Number of turns: {conversation.num_turns}")

    # Save conversation for later
    conversation.save("my_conversation.json")
    print("Conversation saved to my_conversation.json")


if __name__ == "__main__":
    main()
