"""Prompt optimizer for Nexus-LLM.

Provides utilities to improve prompts by adding structure, clarity,
constraints, and few-shot examples.  Also includes a simple quality
measurement heuristic.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimise prompts for better LLM outputs.

    Example::

        opt = PromptOptimizer()
        improved = opt.add_clarity("write code")
        constrained = opt.add_constraints(improved, ["Use Python 3.10+", "No external deps"])
        few_shot = opt.add_examples(constrained, ["Input: sort list → Output: sorted()"])
    """

    # ------------------------------------------------------------------
    # Full optimisation pipeline
    # ------------------------------------------------------------------

    def optimize(self, prompt: str) -> str:
        """Apply a default optimisation pipeline to a prompt.

        The pipeline adds structure, clarity markers, and output
        format instructions.

        Args:
            prompt: The raw prompt string.

        Returns:
            An optimised prompt string.
        """
        result = self.add_clarity(prompt)
        result = self._add_output_format(result)
        return result

    # ------------------------------------------------------------------
    # Individual transformations
    # ------------------------------------------------------------------

    def add_clarity(self, prompt: str) -> str:
        """Add structural clarity to a prompt.

        Transforms a plain prompt into a structured instruction with
        clear task description and expected output section.

        Args:
            prompt: The raw prompt string.

        Returns:
            A clarified, structured prompt.
        """
        prompt = prompt.strip()
        if not prompt:
            return prompt

        # If the prompt already looks structured, return as-is
        if any(marker in prompt.lower() for marker in ["task:", "instruction:", "###"]):
            return prompt

        # Capitalise first letter
        if prompt and prompt[0].islower():
            prompt = prompt[0].upper() + prompt[1:]

        # Ensure it ends with proper punctuation
        if prompt and prompt[-1] not in ".!?:":
            prompt += "."

        # Add structure
        structured = (
            f"### Task\n{prompt}\n\n"
            f"### Instructions\n"
            f"- Be specific and thorough in your response.\n"
            f"- Structure your answer clearly with headings or bullet points where appropriate.\n"
            f"- If the request is ambiguous, state your assumptions.\n"
        )
        return structured

    def add_constraints(self, prompt: str, constraints: List[str]) -> str:
        """Append constraints to a prompt.

        Args:
            prompt: The prompt string.
            constraints: List of constraint strings.

        Returns:
            The prompt with a constraints section appended.
        """
        if not constraints:
            return prompt

        constraint_lines = "\n".join(f"- {c}" for c in constraints)
        section = f"\n### Constraints\n{constraint_lines}\n"
        return prompt.rstrip() + "\n" + section

    def add_examples(self, prompt: str, examples: List[str]) -> str:
        """Add few-shot examples to a prompt.

        Args:
            prompt: The prompt string.
            examples: List of example strings (each should show input/output).

        Returns:
            The prompt with an examples section appended.
        """
        if not examples:
            return prompt

        example_lines = "\n\n".join(
            f"**Example {i + 1}:**\n{ex}" for i, ex in enumerate(examples)
        )
        section = f"\n### Examples\n\n{example_lines}\n"
        return prompt.rstrip() + "\n" + section

    # ------------------------------------------------------------------
    # Quality measurement
    # ------------------------------------------------------------------

    def measure_quality(self, prompt: str, responses: List[str]) -> float:
        """Heuristic quality score for a prompt based on its responses.

        Considers:
        - Response length consistency (lower variance → higher score)
        - Response diversity (distinct responses)
        - Presence of structured markers in responses
        - Prompt clarity (length, structure markers)

        Args:
            prompt: The prompt string.
            responses: List of model responses.

        Returns:
            A quality score in [0, 1].
        """
        if not responses:
            return 0.0

        # --- Prompt quality heuristics ---
        prompt_score = 0.0
        prompt_len = len(prompt.split())

        # Length sweet spot: 10-100 words
        if 10 <= prompt_len <= 100:
            prompt_score += 0.2
        elif prompt_len > 5:
            prompt_score += 0.1

        # Has structural markers
        if any(m in prompt for m in ["###", "Task:", "Instructions:", "- "]):
            prompt_score += 0.15

        # Has variable placeholders
        if re.search(r"\{[a-zA-Z_]", prompt):
            prompt_score += 0.05

        # Ends with clear instruction
        if prompt.rstrip().endswith(".") or prompt.rstrip().endswith(":"):
            prompt_score += 0.1

        # --- Response quality heuristics ---
        response_score = 0.0

        # Length consistency
        lengths = [len(r.split()) for r in responses]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            if avg_len > 0:
                variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
                cv = (variance ** 0.5) / avg_len  # coefficient of variation
                response_score += max(0.0, 0.2 - cv * 0.1)

        # Diversity — distinct responses
        unique_responses = len(set(responses))
        diversity_ratio = unique_responses / len(responses) if responses else 0
        response_score += diversity_ratio * 0.15

        # Structure in responses
        structured_count = sum(
            1 for r in responses if any(m in r for m in ["1.", "- ", "##", "**"])
        )
        structure_ratio = structured_count / len(responses)
        response_score += structure_ratio * 0.15

        total = min(1.0, prompt_score + response_score)
        logger.info(
            "Prompt quality score: %.3f (prompt=%.3f, response=%.3f)",
            total,
            prompt_score,
            response_score,
        )
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_output_format(prompt: str) -> str:
        """Append a default output format section."""
        format_section = (
            "\n### Expected Output Format\n"
            "Provide a clear, well-organised response. "
            "Use headings, bullet points, or numbered lists as appropriate.\n"
        )
        return prompt.rstrip() + "\n" + format_section
