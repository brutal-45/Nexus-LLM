#!/usr/bin/env python3
"""
Custom Prompt Templates - Nexus-LLM
=====================================
Demonstrates how to create and use custom prompt templates
for different tasks and conversational styles.
"""

from nexus_llm import InferenceEngine, Conversation
from nexus_llm.prompts import PromptTemplate, PromptLibrary


def main():
    engine = InferenceEngine(model_name="nexus-7b-chat")

    # --- Define custom prompt templates ---

    # 1. Simple template with variable substitution
    qa_template = PromptTemplate(
        name="qa",
        template="""Answer the following question accurately and concisely.

Question: {question}

Provide your answer in a clear, structured format:
- Direct answer: [your answer]
- Explanation: [brief explanation]
- Confidence: [high/medium/low]""",
    )

    # 2. Few-shot template with examples
    few_shot_template = PromptTemplate(
        name="sentiment",
        template="""Classify the sentiment of the following text.

Examples:
Text: "I love this product! It's amazing."
Sentiment: Positive

Text: "This is terrible. I want a refund."
Sentiment: Negative

Text: "It's okay, nothing special."
Sentiment: Neutral

Now classify:
Text: "{text}"
Sentiment:""",
    )

    # 3. Chain-of-thought template
    cot_template = PromptTemplate(
        name="chain_of_thought",
        template="""Solve the following problem step by step.

Problem: {problem}

Let's think through this:
1. First, I need to understand what is being asked:
2. The key information provided is:
3. My approach will be:
4. Step-by-step solution:
5. Verification:

Final Answer:""",
    )

    # 4. Role-playing template with system context
    roleplay_template = PromptTemplate(
        name="roleplay",
        template="""You are {character_name}, {character_description}.

Your personality traits:
{traits}

Your speaking style:
- {style_point_1}
- {style_point_2}
- {style_point_3}

Always stay in character. Respond to the following as {character_name}:

User: {user_message}
{character_name}:""",
    )

    # --- Register templates in a library ---
    library = PromptLibrary()
    library.register(qa_template)
    library.register(few_shot_template)
    library.register(cot_template)
    library.register(roleplay_template)

    # --- Use templates with the engine ---

    # Using the QA template
    prompt = library.render("qa", question="What is quantum entanglement?")
    print("=== QA Template ===")
    print(prompt)
    print()

    response = engine.generate(prompt)
    print(f"Response: {response.text}\n")

    # Using the few-shot template
    prompt = library.render("sentiment", text="The new update broke all my workflows!")
    print("=== Sentiment Template ===")
    response = engine.generate(prompt)
    print(f"Sentiment: {response.text}\n")

    # Using the chain-of-thought template
    prompt = library.render(
        "chain_of_thought",
        problem="If a train travels at 60 mph and another at 80 mph in the opposite direction, "
                "how long until they are 420 miles apart if they start from the same station?",
    )
    print("=== Chain-of-Thought Template ===")
    response = engine.generate(prompt, max_tokens=512)
    print(f"Solution: {response.text}\n")

    # Using the roleplay template
    prompt = library.render(
        "roleplay",
        character_name="Professor Quantum",
        character_description="a brilliant but eccentric physics professor who explains complex concepts with analogies",
        traits="- Enthusiastic about physics\n- Uses food analogies\n- Gets excited about paradoxes",
        style_point_1="Always uses an exclamation mark at least once per response",
        style_point_2="Relates everything to everyday experiences",
        style_point_3="Occasionally goes on fascinating tangents",
        user_message="Can you explain Schrödinger's cat to me?",
    )
    print("=== Roleplay Template ===")
    response = engine.generate(prompt)
    print(f"Response: {response.text}")

    # Save the library for reuse
    library.save("my_prompt_library.yaml")
    print("\nPrompt library saved to my_prompt_library.yaml")


if __name__ == "__main__":
    main()
