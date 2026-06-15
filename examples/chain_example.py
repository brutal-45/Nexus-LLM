#!/usr/bin/env python3
""" 
Chain Pipelines Example - Nexus-LLM
=====================================
Demonstrates how to create multi-step chain pipelines that
compose multiple LLM operations sequentially.
"""

from nexus_llm import InferenceEngine
from nexus_llm.chains import (
    Chain,
    Step,
    ChainRunner,
    ConditionalStep,
    ParallelStep,
    MapStep,
)


def main():
    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

    # --- Simple sequential chain ---
    print("=" * 60)
    print("Simple Sequential Chain")
    print("=" * 60)

    chain = Chain(
        name="research_summarizer",
        steps=[
            Step(
                name="extract_key_points",
                prompt="Extract 5 key points from the following text:\n\n{input}",
                output_key="key_points",
            ),
            Step(
                name="summarize",
                prompt="Write a concise summary based on these key points:\n\n{key_points}",
                output_key="summary",
            ),
            Step(
                name="translate",
                prompt="Translate the following summary to French:\n\n{summary}",
                output_key="french_summary",
            ),
        ],
    )

    runner = ChainRunner(engine=engine)
    result = runner.run(
        chain,
        input_text="Artificial intelligence has transformed numerous industries, "
                   "from healthcare diagnostics to autonomous vehicles. Machine learning "
                   "algorithms can now detect diseases from medical imaging with "
                   "superhuman accuracy. Self-driving cars use neural networks to "
                   "navigate complex traffic scenarios. Natural language processing "
                   "enables machines to understand and generate human language fluently.",
    )

    print(f"Key Points: {result['key_points'][:200]}...")
    print(f"\nSummary: {result['summary'][:200]}...")
    print(f"\nFrench Summary: {result['french_summary'][:200]}...")

    # --- Chain with conditional branching ---
    print("\n" + "=" * 60)
    print("Conditional Chain")
    print("=" * 60)

    conditional_chain = Chain(
        name="adaptive_response",
        steps=[
            Step(
                name="classify_complexity",
                prompt="Classify the following question as 'simple' or 'complex'. "
                       "Respond with only one word.\n\nQuestion: {input}",
                output_key="complexity",
            ),
            ConditionalStep(
                name="route_by_complexity",
                condition="{complexity}",
                branches={
                    "simple": Step(
                        name="simple_answer",
                        prompt="Answer briefly: {input}",
                        output_key="answer",
                    ),
                    "complex": Chain(
                        name="complex_pipeline",
                        steps=[
                            Step(
                                name="research",
                                prompt="Provide detailed background on: {input}",
                                output_key="research",
                            ),
                            Step(
                                name="detailed_answer",
                                prompt="Based on this research, provide a comprehensive answer:\n\n"
                                       "Question: {input}\nResearch: {research}",
                                output_key="answer",
                            ),
                        ],
                    ),
                },
            ),
        ],
    )

    result = runner.run(conditional_chain, input_text="What is photosynthesis?")
    print(f"Answer: {result['answer'][:200]}...")

    # --- Parallel execution ---
    print("\n" + "=" * 60)
    print("Parallel Chain")
    print("=" * 60)

    parallel_chain = Chain(
        name="multi_perspective",
        steps=[
            ParallelStep(
                name="analyze_from_angles",
                steps=[
                    Step(
                        name="technical_view",
                        prompt="Explain this from a technical perspective: {input}",
                        output_key="technical",
                    ),
                    Step(
                        name="business_view",
                        prompt="Explain this from a business perspective: {input}",
                        output_key="business",
                    ),
                    Step(
                        name="ethical_view",
                        prompt="Explain this from an ethical perspective: {input}",
                        output_key="ethical",
                    ),
                ],
            ),
            Step(
                name="synthesize",
                prompt="Synthesize these three perspectives into one balanced view:\n\n"
                       "Technical: {technical}\nBusiness: {business}\nEthical: {ethical}",
                output_key="synthesis",
            ),
        ],
    )

    result = runner.run(parallel_chain, input_text="The rise of AI in hiring processes")
    print(f"Synthesis: {result['synthesis'][:200]}...")

    # --- Map step for batch processing ---
    print("\n" + "=" * 60)
    print("Map Chain (Batch Processing)")
    print("=" * 60)

    map_chain = Chain(
        name="batch_summarizer",
        steps=[
            MapStep(
                name="summarize_each",
                step=Step(
                    name="summarize_item",
                    prompt="Summarize this in one sentence: {item}",
                    output_key="summary",
                ),
                input_key="items",
                output_key="summaries",
            ),
        ],
    )

    result = runner.run(
        map_chain,
        items=[
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses multiple layers to progressively extract higher-level features.",
        ],
    )

    print("Summaries:")
    for i, summary in enumerate(result["summaries"], 1):
        print(f"  {i}. {summary}")

    # --- Save and load chains ---
    chain.save("./saved_chains/research_summarizer.yaml")
    loaded_chain = Chain.load("./saved_chains/research_summarizer.yaml")
    print(f"\nChain '{loaded_chain.name}' loaded with {len(loaded_chain.steps)} steps")


if __name__ == "__main__":
    main()
