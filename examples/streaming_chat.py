#!/usr/bin/env python3
"""
Streaming Chat Example - Nexus-LLM
====================================
Demonstrates real-time streaming of model responses token by token.
"""

import sys
from nexus_llm import InferenceEngine, Conversation


def stream_callback(token: str, token_id: int, is_final: bool):
    """Called for each token as it is generated."""
    sys.stdout.write(token)
    sys.stdout.flush()
    if is_final:
        print()  # Newline after completion


def main():
    engine = InferenceEngine(
        model_name="nexus-7b-chat",
        device="auto",
    )

    conversation = Conversation(
        system_prompt="You are a creative storytelling assistant."
    )
    conversation.add_user_message(
        "Write a short story about a robot discovering emotions for the first time."
    )

    print("User: Write a short story about a robot discovering emotions for the first time.")
    print("Assistant: ", end="", flush=True)

    # Stream the response token by token
    response = engine.chat_stream(
        conversation,
        callback=stream_callback,
        temperature=0.8,
        top_p=0.95,
        max_tokens=512,
    )

    # After streaming completes, the full response is available
    print(f"\n[Generated {response.token_count} tokens in {response.elapsed_time:.2f}s]")
    print(f"[Throughput: {response.tokens_per_second:.1f} tokens/s]")

    # You can also iterate over the stream manually
    print("\n--- Manual streaming iteration ---")
    conversation.add_user_message("Continue the story.")
    print("User: Continue the story.")
    print("Assistant: ", end="", flush=True)

    stream = engine.chat_stream(conversation, iterate=True)
    full_text = ""
    for chunk in stream:
        sys.stdout.write(chunk.token)
        sys.stdout.flush()
        full_text += chunk.token
        if chunk.is_final:
            print()
            break

    print(f"\nFull response length: {len(full_text)} characters")


if __name__ == "__main__":
    main()
