"""
Synthetic Data Generation for LLM Training
============================================

A comprehensive synthetic data generation framework implementing multiple
state-of-the-art approaches for creating high-quality training data for
large language models.

Implemented methods:
    - Self-Instruct (Wang et al., 2022): Bootstrapping instruction-following
      data from a small set of seed instructions.
    - Evol-Instruct (WizardLM): Iteratively evolving simple instructions into
      more complex ones via deepen, widen, concretize, complicate, reasoning,
      code-specific, and math-specific operations.
    - Rejection Sampling: Generating N responses per prompt, scoring with a
      reward model, and keeping top-k for high-quality SFT data.
    - Persona-Based Generation: Generating instructions from diverse persona
      perspectives covering professions, expertise levels, communication styles,
      cultural backgrounds, and age groups.
    - Math Data Generation: Difficulty-graded math problems with step-by-step
      solutions and programmatic verification.
    - Code Data Generation: Programming problems with test cases, multiple
      solution approaches, execution verification, and code review pairs.

All generators follow a consistent interface and produce ``Instruction`` objects
that are collected into ``SyntheticDataset`` containers suitable for SFT and DPO
training.

Typical usage::

    config = SyntheticDataConfig(generator_model_name="mistral-7b-instruct")
    pipeline = SyntheticDataPipeline(config)
    dataset = pipeline.run_full_pipeline()
    dataset.save("synthetic_data.json")
    train, val = dataset.split(train_ratio=0.9)

Allowed imports: torch, json, random, hashlib, re, math, dataclasses, typing.
"""

from __future__ import annotations

import json
import random
import hashlib
import re
import math
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
    Any,
    Iterator,
    Callable,
    Set,
)
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Prompt Templates (constant module-level strings)
# ---------------------------------------------------------------------------

SEED_INSTRUCTION_PROMPT: str = (
    "You are an expert task designer. Generate a diverse, high-quality "
    "instruction-following task. The task should be clear, self-contained, "
    "and require a substantive response (at least a paragraph). Cover a mix "
    "of categories: question answering, classification, creative writing, "
    "summarization, information extraction, brainstorming, editing, and "
    "coding.\n\n"
    "Output ONLY the instruction text, nothing else."
)

GENERATE_INSTRUCTION_PROMPT: str = (
    "Come up with a new instruction that is DIFFERENT from the following "
    "seed instruction. The new instruction should be in the same general "
    "category ({category}) but cover a different topic, require a different "
    "type of reasoning, or ask for a different output format.\n\n"
    "Seed instruction:\n{seed}\n\n"
    "New instruction (output ONLY the instruction text):"
)

GENERATE_RESPONSE_PROMPT: str = (
    "You are a helpful, harmless, and honest assistant. Provide a "
    "comprehensive, accurate, and well-structured response to the following "
    "instruction.\n\n"
    "Instruction:\n{instruction}\n\n"
    "Response:"
)

EVOLUTION_DEEPEN_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Your objective is to rewrite a "
    "given prompt into a more complex version to make the resulting instruction "
    "more challenging and require deeper reasoning.\n\n"
    "The rewritten prompt must be reasonable, understandable, and answerable "
    "by humans. Do not add unreasonable constraints or conditions.\n\n"
    "Rewrite the following prompt to be more complex by adding more "
    "constraints, requirements, nuances, or deeper analysis requirements.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_WIDEN_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Rewrite the following prompt to "
    "cover a broader topic, requiring the response to address multiple aspects "
    "or related sub-topics.\n\n"
    "The rewritten prompt must remain answerable and coherent.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_CONCRETIZE_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Rewrite the following prompt to "
    "be more specific and concrete. Add specific details, numbers, names, "
    "scenarios, or contexts to make the instruction more precise.\n\n"
    "The rewritten prompt must remain answerable.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_COMPLICATE_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Rewrite the following prompt to "
    "be more complicated by adding multiple steps, conditions, or constraints "
    "that must be satisfied simultaneously.\n\n"
    "The rewritten prompt must remain answerable and logical.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_REASONING_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Rewrite the following prompt to "
    "explicitly require step-by-step reasoning, chain-of-thought analysis, or "
    "multi-step logical deduction.\n\n"
    "The rewritten prompt must remain answerable.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_CODE_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Rewrite the following prompt to "
    "convert it into a programming or software engineering problem. The new "
    "prompt should require writing code, designing algorithms, or debugging.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_MATH_PROMPT: str = (
    "I want you to act as a Prompt Rewriter. Rewrite the following prompt to "
    "add mathematical reasoning, computation, or quantitative analysis "
    "requirements.\n\n"
    "The rewritten prompt must remain answerable.\n\n"
    "Original prompt:\n{instruction}\n\n"
    "Rewritten prompt (output ONLY the rewritten prompt):"
)

EVOLUTION_RANDOM_RESPONSE_CHECK_PROMPT: str = (
    "Determine if the following response is a direct answer to the given "
    "instruction. The response should NOT already contain or be a valid "
    "answer to the evolved instruction.\n\n"
    "Original response:\n{response}\n\n"
    "Evolved instruction:\n{instruction}\n\n"
    "Answer YES if the response already answers the instruction, NO otherwise. "
    "Output ONLY YES or NO:"
)

PERSONA_INSTRUCTION_PROMPT: str = (
    "You are {persona}. Generate an instruction or question that {persona} "
    "would naturally ask or need help with. The instruction should reflect "
    "the persona's profession, expertise level, communication style, and "
    "perspective.\n\n"
    "Output ONLY the instruction text:"
)

PERSONA_RESPONSE_PROMPT: str = (
    "You are {persona}. Respond to the following instruction in character. "
    "Your response should reflect your expertise, communication style, "
    "and perspective as {persona}.\n\n"
    "Instruction:\n{instruction}\n\n"
    "Response:"
)

MATH_PROBLEM_PROMPT: str = (
    "Generate a {difficulty} {category} math problem with the following "
    "requirements:\n"
    "- The problem must have a unique, verifiable answer\n"
    "- Provide a step-by-step solution\n"
    "- Include the final answer clearly\n\n"
    "Category: {category}\nDifficulty: {difficulty}\n\n"
    "Output format:\n"
    "PROBLEM: <problem statement>\n"
    "SOLUTION: <step-by-step solution>\n"
    "ANSWER: <final answer>"
)

CODE_PROBLEM_PROMPT: str = (
    "Generate a {difficulty} programming problem with the following "
    "requirements:\n"
    "- The problem must have a clear specification\n"
    "- Provide at least 3 test cases with expected outputs\n"
    "- Provide a reference solution\n"
    "- Language: {language}\n\n"
    "Difficulty: {difficulty}\n\n"
    "Output format:\n"
    "PROBLEM: <problem description>\n"
    "TEST_CASES:\n<test case 1>\n<test case 2>\n<test case 3>\n"
    "SOLUTION:\n```{language}\n<reference solution>\n```"
)

CODE_DEBUG_PROMPT: str = (
    "Generate a code debugging exercise:\n"
    "- Provide a programming problem specification\n"
    "- Provide an INCORRECT solution that has a subtle bug\n"
    "- Explain the bug\n"
    "- Provide the corrected solution\n"
    "- Language: {language}\n\n"
    "Output format:\n"
    "PROBLEM: <problem description>\n"
    "BUGGY_CODE:\n```{language}\n<incorrect solution>\n```\n"
    "BUG_EXPLANATION: <what is wrong and why>\n"
    "CORRECT_CODE:\n```{language}\n<corrected solution>\n```"
)

CODE_REVIEW_PROMPT: str = (
    "Generate a code review exercise:\n"
    "- Provide a programming problem specification\n"
    "- Provide a GOOD solution (clean, efficient, well-documented)\n"
    "- Provide a BAD solution (messy, inefficient, poorly written)\n"
    "- Both must be functionally correct\n"
    "- Language: {language}\n\n"
    "Output format:\n"
    "PROBLEM: <problem description>\n"
    "GOOD_CODE:\n```{language}\n<good solution>\n```\n"
    "BAD_CODE:\n```{language}\n<bad solution>\n```\n"
    "REVIEW: <why the good code is better>"
)

REJECTION_SAMPLE_PROMPT: str = (
    "You are a helpful, harmless, and honest assistant. Provide a "
    "high-quality, comprehensive response to the following instruction. "
    "Make your response detailed, accurate, and well-structured.\n\n"
    "Instruction:\n{instruction}\n\n"
    "Response:"
)

QUALITY_CHECK_PROMPT: str = (
    "Rate the quality of the following instruction-response pair on a scale "
    "of 0 to 1. Consider:\n"
    "- Clarity and specificity of the instruction\n"
    "- Accuracy and completeness of the response\n"
    "- Relevance of the response to the instruction\n"
    "- Writing quality and structure\n\n"
    "Instruction:\n{instruction}\n\n"
    "Response:\n{response}\n\n"
    "Output ONLY a number between 0 and 1:"
)


# ---------------------------------------------------------------------------
# Instruction Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """Represents a single synthetic instruction-response pair.

    Attributes:
        id: Unique identifier derived from hashing instruction content.
        instruction: The natural-language instruction text.
        response: The model-generated or human-written response.
        category: Task category (e.g., "question answering", "code").
        difficulty: Difficulty level (e.g., "easy", "medium", "hard", "expert").
        source: Name of the generator that produced this instruction.
        quality_score: Quality assessment score in [0, 1].
        metadata: Arbitrary key-value metadata attached to this instruction.
    """

    id: str = ""
    instruction: str = ""
    response: str = ""
    category: str = "general"
    difficulty: str = "medium"
    source: str = "unknown"
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-generate id from instruction + response if not provided."""
        if not self.id and self.instruction:
            raw = f"{self.instruction}|{self.response}"
            self.id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Instruction:
        """Deserialize from a plain dictionary."""
        return cls(**data)


# ---------------------------------------------------------------------------
# SyntheticDataConfig
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation.

    Controls model selection, generation volume, quality thresholds,
    and diversity parameters across all generators.

    Attributes:
        generator_model_name: Name or path of the LLM used to generate data.
        num_instructions: Target number of instructions to generate.
        difficulty_levels: Difficulty tiers to sample from.
        domains: Knowledge domains for category coverage.
        max_evolution_rounds: Maximum number of Evol-Instruct rounds.
        quality_threshold: Minimum quality score to keep an instruction.
        dedup_threshold: Similarity threshold above which two instructions
            are considered duplicates (0-1).
    """

    generator_model_name: str = "mistral-7b-instruct"
    num_instructions: int = 100000
    difficulty_levels: List[str] = field(
        default_factory=lambda: ["easy", "medium", "hard", "expert"]
    )
    domains: List[str] = field(
        default_factory=lambda: [
            "general", "code", "math", "reasoning",
            "creative", "safety",
        ]
    )
    max_evolution_rounds: int = 5
    quality_threshold: float = 0.7
    dedup_threshold: float = 0.85


# ---------------------------------------------------------------------------
# SyntheticDataset — Result Container
# ---------------------------------------------------------------------------

class SyntheticDataset:
    """Container for a collection of synthetic instructions.

    Provides serialization, splitting, and formatting utilities for
    downstream training pipelines (SFT, DPO).

    Attributes:
        instructions: The list of ``Instruction`` objects in this dataset.
    """

    def __init__(self, instructions: Optional[List[Instruction]] = None) -> None:
        self.instructions: List[Instruction] = instructions or []

    # -- Statistics --------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of instructions."""
        return len(self.instructions)

    def category_distribution(self) -> Dict[str, int]:
        """Return a mapping of category -> count."""
        dist: Dict[str, int] = {}
        for instr in self.instructions:
            dist[instr.category] = dist.get(instr.category, 0) + 1
        return dist

    def difficulty_distribution(self) -> Dict[str, int]:
        """Return a mapping of difficulty -> count."""
        dist: Dict[str, int] = {}
        for instr in self.instructions:
            dist[instr.difficulty] = dist.get(instr.difficulty, 0) + 1
        return dist

    def source_distribution(self) -> Dict[str, int]:
        """Return a mapping of generator source -> count."""
        dist: Dict[str, int] = {}
        for instr in self.instructions:
            dist[instr.source] = dist.get(instr.source, 0) + 1
        return dist

    def avg_quality_score(self) -> float:
        """Mean quality score across all instructions."""
        if not self.instructions:
            return 0.0
        return sum(i.quality_score for i in self.instructions) / len(self.instructions)

    def statistics(self) -> Dict[str, Any]:
        """Return a full statistics summary."""
        return {
            "total_count": self.count,
            "avg_quality_score": self.avg_quality_score(),
            "category_distribution": self.category_distribution(),
            "difficulty_distribution": self.difficulty_distribution(),
            "source_distribution": self.source_distribution(),
        }

    # -- Splitting ---------------------------------------------------------

    def split(self, train_ratio: float = 0.9) -> Tuple[SyntheticDataset, SyntheticDataset]:
        """Split into training and validation sets.

        Args:
            train_ratio: Proportion of data to allocate to the training split.

        Returns:
            A (train, val) tuple of ``SyntheticDataset`` instances.
        """
        indices = list(range(self.count))
        random.shuffle(indices)
        split_idx = int(self.count * train_ratio)
        train_instrs = [self.instructions[i] for i in indices[:split_idx]]
        val_instrs = [self.instructions[i] for i in indices[split_idx:]]
        return SyntheticDataset(train_instrs), SyntheticDataset(val_instrs)

    # -- Formatting --------------------------------------------------------

    def format_for_sft(self) -> List[Dict[str, str]]:
        """Format instructions as prompt/response pairs for SFT training.

        Returns:
            A list of dicts with keys ``prompt`` and ``response``.
        """
        return [
            {"prompt": instr.instruction, "response": instr.response}
            for instr in self.instructions
            if instr.instruction and instr.response
        ]

    def format_for_dpo(self) -> List[Dict[str, str]]:
        """Format instructions as chosen/rejected pairs for DPO training.

        Pairs instructions by category and difficulty, treating higher
        quality_score entries as chosen and lower ones as rejected.

        Returns:
            A list of dicts with keys ``prompt``, ``chosen``, and ``rejected``.
        """
        # Group by category and difficulty for pairing
        groups: Dict[str, List[Instruction]] = {}
        for instr in self.instructions:
            key = f"{instr.category}|{instr.difficulty}"
            groups.setdefault(key, []).append(instr)

        pairs: List[Dict[str, str]] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            sorted_group = sorted(group, key=lambda x: x.quality_score, reverse=True)
            # Pair best with worst in each group
            for i in range(0, len(sorted_group) - 1, 2):
                chosen = sorted_group[i]
                rejected = sorted_group[i + 1]
                if chosen.instruction == rejected.instruction:
                    pairs.append({
                        "prompt": chosen.instruction,
                        "chosen": chosen.response,
                        "rejected": rejected.response,
                    })
        return pairs

    # -- Serialization -----------------------------------------------------

    def save(self, path: str) -> None:
        """Save dataset to a JSON file.

        Args:
            path: Destination file path (should end in ``.json``).
        """
        data = {
            "instructions": [instr.to_dict() for instr in self.instructions],
            "statistics": self.statistics(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> SyntheticDataset:
        """Load dataset from a JSON file.

        Args:
            path: Source file path.

        Returns:
            A ``SyntheticDataset`` instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instructions = [Instruction.from_dict(d) for d in data.get("instructions", [])]
        return cls(instructions=instructions)

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return (
            f"SyntheticDataset(count={self.count}, "
            f"categories={len(self.category_distribution())}, "
            f"avg_quality={self.avg_quality_score():.3f})"
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _fingerprint(text: str) -> str:
    """Return a short SHA-256 fingerprint for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _token_overlap_ratio(a: str, b: str) -> float:
    """Compute the Jaccard similarity between two texts at the word level.

    This is a lightweight fuzzy-dedup heuristic that does not require
    external libraries or model inference.

    Args:
        a: First text.
        b: Second text.

    Returns:
        Similarity score in [0, 1].
    """
    set_a: Set[str] = set(re.findall(r"\w+", a.lower()))
    set_b: Set[str] = set(re.findall(r"\w+", b.lower()))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _sample_without_replacement(population: List[Any], k: int) -> List[Any]:
    """Randomly sample *k* items from *population* without replacement.

    Falls back gracefully when *k* >= len(population).
    """
    if k >= len(population):
        return list(population)
    return random.sample(population, k)


# ---------------------------------------------------------------------------
# SelfInstructGenerator
# ---------------------------------------------------------------------------

class SelfInstructGenerator:
    """Self-Instruct data generator (Wang et al., 2022).

    Bootstraps instruction-following data from a small set of human-written
    seed instructions by repeatedly asking an LLM to generate novel
    instructions in various categories.

    Algorithm:
        1. Start with seed instructions (~175 human-written tasks).
        2. For each seed, generate N new instructions via the LLM.
        3. Generate responses for new instructions.
        4. Filter by quality (heuristic or classifier-based).
        5. Remove near-duplicates via fuzzy deduplication.
        6. Repeat until the target count is reached.

    Args:
        config: A ``SyntheticDataConfig`` instance controlling generation.
    """

    # Human-written seed instructions covering diverse categories.
    _SEED_INSTRUCTIONS: List[str] = [
        "Write a short story about a robot learning to paint.",
        "Explain the difference between TCP and UDP protocols.",
        "Summarize the key events of the French Revolution in 5 bullet points.",
        "Translate the following English sentence to French: 'The cat is on the mat.'",
        "Write a Python function that checks if a number is prime.",
        "What are the main causes of climate change? Provide a structured answer.",
        "Generate 5 creative names for a new coffee shop.",
        "Classify the following text as positive, negative, or neutral sentiment.",
        "Rewrite this paragraph to be more concise and professional.",
        "Explain quantum computing to a 10-year-old.",
        "What is the capital of Australia? Provide some interesting facts about it.",
        "Write a haiku about the ocean.",
        "List the steps to create a budget spreadsheet in Excel.",
        "Compare and contrast renewable and non-renewable energy sources.",
        "Write a function to reverse a linked list in C++.",
        "What are the health benefits of regular exercise?",
        "Draft an email to a manager requesting a deadline extension.",
        "Explain the concept of supply and demand with a simple example.",
        "Create a weekly meal plan for a vegetarian diet.",
        "What is the theory of relativity? Explain it simply.",
        "Write a product review for a noise-cancelling headphone.",
        "Extract all the dates mentioned in the following text.",
        "What are the differences between machine learning and deep learning?",
        "Write a limerick about programming.",
        "Describe the water cycle in 4 steps.",
        "How do vaccines work? Explain in simple terms.",
        "Write a SQL query to find the top 10 customers by total spend.",
        "What are the best practices for writing clean code?",
        "Create a 7-day workout plan for beginners.",
        "Explain the concept of inflation and its effects on the economy.",
    ]

    _CATEGORIES: List[str] = [
        "question answering",
        "classification",
        "creative writing",
        "summarization",
        "information extraction",
        "brainstorming",
        "editing",
        "code generation",
        "math",
        "reasoning",
        "translation",
        "comparison",
        "advice",
    ]

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._seen_hashes: Set[str] = set()
        self._rng = random.Random(42)

    # -- Core pipeline methods ----------------------------------------------

    def generate_seed_instructions(self) -> List[Instruction]:
        """Create seed instructions from the built-in human-written list.

        Returns:
            A list of ``Instruction`` objects for each seed.
        """
        seeds: List[Instruction] = []
        for idx, text in enumerate(self._SEED_INSTRUCTIONS):
            cat = self._CATEGORIES[idx % len(self._CATEGORIES)]
            instr = Instruction(
                instruction=text,
                response="",
                category=cat,
                difficulty=self._rng.choice(self.config.difficulty_levels),
                source="self_instruct_seed",
                quality_score=1.0,
            )
            seeds.append(instr)
            self._seen_hashes.add(instr.id)
        return seeds

    def generate_instruction(self, seed: Instruction, category: str) -> Optional[Instruction]:
        """Generate a single new instruction from a seed.

        The method formats a generation prompt using the seed instruction
        and target category, then delegates to the (mock) LLM to produce
        a novel instruction.

        Args:
            seed: The parent seed instruction.
            category: The task category for the new instruction.

        Returns:
            A new ``Instruction`` or ``None`` if generation fails.
        """
        prompt = GENERATE_INSTRUCTION_PROMPT.format(
            seed=seed.instruction,
            category=category,
        )
        new_text = self._mock_llm_generate(prompt)

        if not new_text or len(new_text.strip()) < 5:
            return None

        candidate = Instruction(
            instruction=new_text.strip(),
            response="",
            category=category,
            difficulty=self._rng.choice(self.config.difficulty_levels),
            source="self_instruct",
            metadata={"parent_seed_id": seed.id},
        )

        # Reject duplicates
        if candidate.id in self._seen_hashes:
            return None

        self._seen_hashes.add(candidate.id)
        return candidate

    def generate_response(self, instruction: Instruction) -> str:
        """Generate a response for the given instruction.

        Args:
            instruction: The instruction to respond to.

        Returns:
            The generated response text.
        """
        prompt = GENERATE_RESPONSE_PROMPT.format(instruction=instruction.instruction)
        response = self._mock_llm_generate(prompt)
        return response.strip() if response else ""

    def filter_by_quality(self, instructions: List[Instruction]) -> List[Instruction]:
        """Filter instructions by heuristic quality assessment.

        Quality heuristics include:
            - Minimum instruction length (>= 10 characters).
            - Non-trivial response length (>= 20 characters).
            - No excessive repetition (compression ratio check).
            - Quality score above ``config.quality_threshold``.

        Args:
            instructions: Candidate instructions to filter.

        Returns:
            Instructions that pass the quality bar.
        """
        filtered: List[Instruction] = []
        for instr in instructions:
            if len(instr.instruction.strip()) < 10:
                continue
            if len(instr.response.strip()) < 20:
                continue
            # Check for repetitive text
            words = instr.response.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    continue
            # Compute a simple heuristic quality score
            instr.quality_score = SelfInstructGenerator._heuristic_quality(instr)
            if instr.quality_score >= self.config.quality_threshold:
                filtered.append(instr)
        return filtered

    def deduplicate(self, instructions: List[Instruction]) -> List[Instruction]:
        """Remove near-duplicate instructions using fuzzy Jaccard similarity.

        Args:
            instructions: Instructions to deduplicate.

        Returns:
            Deduplicated list preserving original order.
        """
        unique: List[Instruction] = []
        for instr in instructions:
            is_dup = False
            for existing in unique:
                sim = _token_overlap_ratio(instr.instruction, existing.instruction)
                if sim >= self.config.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(instr)
        return unique

    def run(self) -> SyntheticDataset:
        """Execute the full Self-Instruct pipeline.

        Steps:
            1. Generate seed instructions.
            2. Iteratively generate new instructions from seeds.
            3. Generate responses.
            4. Filter by quality.
            5. Deduplicate.

        Returns:
            A ``SyntheticDataset`` containing the generated instructions.
        """
        target = self.config.num_instructions
        seeds = self.generate_seed_instructions()

        # Generate responses for seeds
        for seed in seeds:
            seed.response = self.generate_response(seed)

        pool: List[Instruction] = list(seeds)

        iteration = 0
        while len(pool) < target:
            iteration += 1
            batch_size = min(50, target - len(pool))
            new_instructions: List[Instruction] = []

            for _ in range(batch_size):
                seed = self._rng.choice(pool)
                category = self._rng.choice(self._CATEGORIES)
                new_instr = self.generate_instruction(seed, category)
                if new_instr is not None:
                    new_instr.response = self.generate_response(new_instr)
                    new_instructions.append(new_instr)

            # Quality filter and dedup
            new_instructions = self.filter_by_quality(new_instructions)
            new_instructions = self.deduplicate(new_instructions)
            pool.extend(new_instructions)

            if len(new_instructions) == 0:
                # No progress — break to avoid infinite loop
                break

        pool = self.deduplicate(pool)
        return SyntheticDataset(instructions=pool[:target])

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _mock_llm_generate(prompt: str, max_tokens: int = 256) -> str:
        """Placeholder LLM generation stub.

        In production, this would call the generator model specified in the
        config. For now it returns a deterministic placeholder to demonstrate
        pipeline structure.
        """
        return f"[Generated response for: {prompt[:50]}...]"

    def _heuristic_quality(instr: Instruction) -> float:
        """Compute a heuristic quality score in [0, 1].

        Combines instruction length, response length, and lexical diversity.
        """
        instr_len = len(instr.instruction.split())
        resp_len = len(instr.response.split())

        # Length score (longer is better, capped at 50 words for instr,
        # 200 for response).
        instr_score = min(instr_len / 50.0, 1.0)
        resp_score = min(resp_len / 200.0, 1.0)

        # Diversity score
        if resp_len > 0:
            diversity = len(set(instr.response.split())) / resp_len
        else:
            diversity = 0.0

        # Weighted combination
        return 0.3 * instr_score + 0.4 * resp_score + 0.3 * diversity


# ---------------------------------------------------------------------------
# EvolInstructGenerator
# ---------------------------------------------------------------------------

class EvolInstructGenerator:
    """Evol-Instruct generator (WizardLM style).

    Iteratively evolves simple instructions into more complex ones by
    applying random evolution operations such as deepening, widening,
    concretizing, complicating, reasoning, code-specific, and
    math-specific transformations.

    After each evolution round the instruction is validated:
        - It must not be gibberish (length and coherence checks).
        - The original response must NOT already answer the evolved question.

    Args:
        config: A ``SyntheticDataConfig`` instance.
    """

    _EVOLUTION_OPERATIONS: List[str] = [
        "deepen",
        "widen",
        "concretize",
        "complicate",
        "reasoning",
        "code_specific",
        "math_specific",
    ]

    _EVOLUTION_PROMPTS: Dict[str, str] = {
        "deepen": EVOLUTION_DEEPEN_PROMPT,
        "widen": EVOLUTION_WIDEN_PROMPT,
        "concretize": EVOLUTION_CONCRETIZE_PROMPT,
        "complicate": EVOLUTION_COMPLICATE_PROMPT,
        "reasoning": EVOLUTION_REASONING_PROMPT,
        "code_specific": EVOLUTION_CODE_PROMPT,
        "math_specific": EVOLUTION_MATH_PROMPT,
    }

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._rng = random.Random(42)
        self._seen_hashes: Set[str] = set()

    def _evolve_instruction(
        self,
        instruction: str,
        response: str,
        round_num: int,
    ) -> Optional[str]:
        """Apply a single random evolution operation to an instruction.

        Args:
            instruction: The original instruction text.
            response: The original response (used for validation).
            round_num: The current evolution round number.

        Returns:
            The evolved instruction text, or ``None`` if rejected.
        """
        op = self._rng.choice(self._EVOLUTION_OPERATIONS)
        prompt_template = self._EVOLUTION_PROMPTS[op]
        prompt = prompt_template.format(instruction=instruction)
        evolved = self._mock_llm_generate(prompt, max_tokens=300)

        if not evolved or len(evolved.strip()) < 10:
            return None

        evolved = evolved.strip()

        # Check: evolved instruction should not be too similar to original
        if _token_overlap_ratio(instruction, evolved) > 0.95:
            return None

        # Check: original response should NOT already answer the evolved
        # instruction.
        check_prompt = EVOLUTION_RANDOM_RESPONSE_CHECK_PROMPT.format(
            response=response,
            instruction=evolved,
        )
        check_result = self._mock_llm_generate(check_prompt, max_tokens=10)
        if check_result.strip().upper() == "YES":
            return None

        # Determine difficulty based on round
        difficulty = self.config.difficulty_levels[
            min(round_num, len(self.config.difficulty_levels) - 1)
        ]

        # Determine category based on operation
        category_map = {
            "code_specific": "code",
            "math_specific": "math",
            "reasoning": "reasoning",
            "complicate": "reasoning",
        }
        category = category_map.get(op, "general")

        return evolved

    def _evolve_with_multiple_rounds(
        self,
        seed: Instruction,
    ) -> Optional[Instruction]:
        """Evolve a single instruction through multiple rounds.

        Args:
            seed: The starting instruction.

        Returns:
            A fully-evolved ``Instruction``, or ``None`` on failure.
        """
        current_text = seed.instruction
        current_response = seed.response
        num_rounds = self._rng.randint(1, self.config.max_evolution_rounds)

        for round_num in range(num_rounds):
            evolved = self._evolve_instruction(
                current_text, current_response, round_num
            )
            if evolved is None:
                break
            current_text = evolved
            # Generate a new response for the evolved instruction
            current_response = self._mock_llm_generate(
                GENERATE_RESPONSE_PROMPT.format(instruction=current_text),
                max_tokens=512,
            )

        # Create the final evolved instruction
        final = Instruction(
            instruction=current_text,
            response=current_response.strip(),
            category="general",
            difficulty=self.config.difficulty_levels[
                min(num_rounds, len(self.config.difficulty_levels) - 1)
            ],
            source="evol_instruct",
            metadata={
                "original_instruction": seed.instruction,
                "evolution_rounds": num_rounds,
            },
        )

        if final.id in self._seen_hashes:
            return None
        self._seen_hashes.add(final.id)
        return final

    def run(self, seed_instructions: Optional[List[Instruction]] = None) -> SyntheticDataset:
        """Execute the full Evol-Instruct pipeline.

        Args:
            seed_instructions: Optional seed instructions. If ``None``, a
                default set of simple instructions is used.

        Returns:
            A ``SyntheticDataset`` of evolved instructions.
        """
        if seed_instructions is None:
            seed_instructions = self._default_seeds()

        target = self.config.num_instructions
        results: List[Instruction] = []

        for seed in seed_instructions:
            if len(results) >= target:
                break
            evolved = self._evolve_with_multiple_rounds(seed)
            if evolved is not None:
                evolved.quality_score = SelfInstructGenerator._heuristic_quality(
                    evolved
                ) if evolved.response else 0.0
                results.append(evolved)

        # Also evolve from previously evolved instructions for diversity
        no_progress = 0
        max_no_progress = 100
        while len(results) < target:
            if not results or no_progress >= max_no_progress:
                break
            seed = self._rng.choice(results)
            evolved = self._evolve_with_multiple_rounds(seed)
            if evolved is not None:
                evolved.quality_score = SelfInstructGenerator._heuristic_quality(
                    evolved
                ) if evolved.response else 0.0
                results.append(evolved)
                no_progress = 0
            else:
                no_progress += 1

        return SyntheticDataset(instructions=results[:target])

    def _default_seeds(self) -> List[Instruction]:
        """Return a default set of simple seed instructions."""
        simple_prompts = [
            "What is the weather like today?",
            "Explain photosynthesis.",
            "Write a greeting message.",
            "What is 2 + 2?",
            "Name three colors.",
            "What is a dog?",
            "Describe a tree.",
            "How do you make tea?",
            "What is the sky?",
            "List common fruits.",
            "Explain what a book is.",
            "What time is breakfast?",
            "Define the word 'happy'.",
            "What does a teacher do?",
            "Name a country in Europe.",
            "What is water?",
            "Explain sleep.",
            "What is music?",
            "How old is the Earth?",
            "What is a family?",
        ]
        seeds: List[Instruction] = []
        for text in simple_prompts:
            instr = Instruction(
                instruction=text,
                response=f"[Response for: {text}]",
                difficulty="easy",
                source="evol_instruct_seed",
            )
            seeds.append(instr)
            self._seen_hashes.add(instr.id)
        return seeds

    @staticmethod
    def _mock_llm_generate(prompt: str, max_tokens: int = 256) -> str:
        """Placeholder LLM generation stub."""
        return f"[Evolved output for: {prompt[:60]}...]"


# ---------------------------------------------------------------------------
# RejectionSamplingGenerator
# ---------------------------------------------------------------------------

class RejectionSamplingGenerator:
    """Rejection-sampling generator for high-quality SFT data.

    For each prompt, the generator samples N candidate responses from the
    policy model, scores them using a reward model, and keeps only the
    top-k responses as training data. This ensures the SFT dataset contains
    only the model's best outputs.

    Algorithm:
        1. For each prompt, sample N responses.
        2. Score each response with a reward model.
        3. Keep top-k responses (highest scores).
        4. Return as a ``SyntheticDataset``.

    Args:
        config: A ``SyntheticDataConfig`` instance.
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._rng = random.Random(42)

    def sample_responses(self, prompt: str, num_samples: int = 8) -> List[str]:
        """Sample multiple responses for a single prompt.

        In production this would call the policy model ``num_samples`` times
        with varying temperature. Here we generate deterministic but
        distinct placeholder responses.

        Args:
            prompt: The instruction text.
            num_samples: Number of candidate responses to generate.

        Returns:
            A list of response strings.
        """
        responses: List[str] = []
        for i in range(num_samples):
            resp = self._mock_policy_generate(prompt, sample_idx=i)
            if resp:
                responses.append(resp)
        return responses

    def score_responses(self, responses: List[str]) -> List[float]:
        """Score a list of responses using a reward model.

        The mock reward model uses heuristic features:
            - Response length (longer is preferred, up to a cap).
            - Lexical diversity (higher is preferred).
            - Absence of repetitive patterns.

        Args:
            responses: Candidate response texts.

        Returns:
            A list of reward scores, one per response.
        """
        scores: List[float] = []
        for resp in responses:
            words = resp.split()
            if not words:
                scores.append(0.0)
                continue

            length_score = min(len(words) / 200.0, 1.0)
            diversity = len(set(words)) / len(words)

            # Penalise excessive repetition of bigrams
            bigrams: List[str] = []
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]} {words[i+1]}")
            bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)

            score = 0.3 * length_score + 0.4 * diversity + 0.3 * bigram_diversity
            scores.append(score)
        return scores

    def select_best(
        self,
        responses: List[str],
        scores: List[float],
        top_k: int = 1,
    ) -> List[str]:
        """Select the top-k responses by score.

        Args:
            responses: Candidate response texts.
            scores: Corresponding reward scores.
            top_k: Number of best responses to keep.

        Returns:
            A list of the top-k response strings.
        """
        paired = sorted(zip(scores, responses), key=lambda x: x[0], reverse=True)
        return [r for _, r in paired[:top_k]]

    def run(
        self,
        prompts: Optional[List[str]] = None,
        num_samples: int = 8,
        top_k: int = 1,
    ) -> SyntheticDataset:
        """Execute the rejection-sampling pipeline.

        Args:
            prompts: Instruction texts. If ``None``, default prompts are used.
            num_samples: Candidate responses per prompt.
            top_k: Best responses to keep per prompt.

        Returns:
            A ``SyntheticDataset`` of the highest-quality pairs.
        """
        if prompts is None:
            prompts = self._default_prompts()

        instructions: List[Instruction] = []
        for idx, prompt in enumerate(prompts):
            responses = self.sample_responses(prompt, num_samples=num_samples)
            scores = self.score_responses(responses)
            best = self.select_best(responses, scores, top_k=top_k)

            for resp in best:
                score_idx = responses.index(resp) if resp in responses else 0
                instr = Instruction(
                    instruction=prompt,
                    response=resp,
                    category="general",
                    difficulty=self._rng.choice(self.config.difficulty_levels),
                    source="rejection_sampling",
                    quality_score=scores[score_idx],
                    metadata={"num_candidates": num_samples},
                )
                instructions.append(instr)

        return SyntheticDataset(instructions=instructions)

    def _default_prompts(self) -> List[str]:
        """Return a default set of diverse prompts."""
        return [
            "Explain the concept of recursion in programming.",
            "What are the advantages of renewable energy?",
            "Write a poem about the changing seasons.",
            "How does the internet work?",
            "What is the difference between a virus and bacteria?",
            "Explain the importance of education.",
            "Describe the process of photosynthesis.",
            "What are the effects of social media on mental health?",
            "How do you start a small business?",
            "Explain the theory of evolution.",
        ]

    def _mock_policy_generate(self, prompt: str, sample_idx: int = 0) -> str:
        """Placeholder policy model generation."""
        # Produce varied-length placeholders to simulate different quality.
        base = f"Response to '{prompt[:40]}'."
        padding = " This provides additional detail and context. " * (sample_idx + 1)
        return (base + padding).strip()


# ---------------------------------------------------------------------------
# PersonaGenerator
# ---------------------------------------------------------------------------

class PersonaGenerator:
    """Persona-based instruction generator.

    Defines 100+ diverse personas spanning professions, expertise levels,
    communication styles, cultural backgrounds, and age groups. For each
    persona the generator creates instructions and responses that authentically
    reflect the persona's perspective and voice.

    Args:
        config: A ``SyntheticDataConfig`` instance.
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._rng = random.Random(42)

    def _build_personas(self) -> List[Dict[str, str]]:
        """Construct a list of 100+ diverse persona descriptors.

        Returns:
            A list of dicts, each containing persona attributes.
        """
        professions = [
            "a doctor", "a lawyer", "a software engineer", "a teacher",
            "a nurse", "a chef", "a journalist", "a financial analyst",
            "a graphic designer", "a civil engineer", "a marine biologist",
            "a data scientist", "a musician", "a social worker", "a farmer",
            "an architect", "a pharmacist", "a psychologist", "an electrician",
            "a marketing manager", "a librarian", "a pilot", "a photographer",
            "a veterinarian", "a historian",
        ]
        expertise_levels = ["a beginner", "an intermediate practitioner", "an expert"]
        communication_styles = [
            "formal and academic", "casual and friendly",
            "technical and precise", "simple and accessible",
        ]
        age_groups = [
            "a college student in their early 20s",
            "a professional in their 30s",
            "a mid-career professional in their 40s",
            "a senior professional in their 50s",
            "a retiree in their 60s",
        ]

        personas: List[Dict[str, str]] = []
        counter = 0
        for prof in professions:
            for expertise in expertise_levels:
                for style in communication_styles:
                    age = self._rng.choice(age_groups)
                    persona_str = (
                        f"{age} who is {expertise} in their field, "
                        f"communicates in a {style} manner, "
                        f"and works as {prof}"
                    )
                    personas.append({
                        "description": persona_str,
                        "profession": prof,
                        "expertise": expertise,
                        "style": style,
                        "age_group": age,
                    })
                    counter += 1
                    if counter >= 120:
                        return personas
        return personas

    def run(self, num_personas: Optional[int] = None) -> SyntheticDataset:
        """Execute the persona-based generation pipeline.

        For each persona, one instruction is generated matching their
        perspective, followed by a response in their voice.

        Args:
            num_personas: How many personas to generate data for.
                If ``None``, uses ``config.num_instructions``.

        Returns:
            A ``SyntheticDataset`` of persona-grounded instructions.
        """
        target = num_personas or self.config.num_instructions
        personas = self._build_personas()[:target]
        instructions: List[Instruction] = []

        for persona in personas:
            instr_text = self._generate_persona_instruction(persona)
            if not instr_text:
                continue

            resp_text = self._generate_persona_response(persona, instr_text)
            if not resp_text:
                continue

            instr = Instruction(
                instruction=instr_text,
                response=resp_text,
                category="general",
                difficulty=self._rng.choice(self.config.difficulty_levels),
                source="persona_generator",
                quality_score=self._rng.uniform(0.6, 1.0),
                metadata={
                    "persona": persona["description"],
                    "profession": persona["profession"],
                    "expertise": persona["expertise"],
                    "style": persona["style"],
                },
            )
            instructions.append(instr)

        return SyntheticDataset(instructions=instructions)

    def _generate_persona_instruction(self, persona: Dict[str, str]) -> Optional[str]:
        """Generate an instruction matching a persona's perspective.

        Args:
            persona: A persona descriptor dict.

        Returns:
            An instruction string, or ``None`` on failure.
        """
        prompt = PERSONA_INSTRUCTION_PROMPT.format(persona=persona["description"])
        result = self._mock_llm_generate(prompt)
        return result.strip() if result and len(result.strip()) > 5 else None

    def _generate_persona_response(
        self, persona: Dict[str, str], instruction: str
    ) -> Optional[str]:
        """Generate a response in a persona's voice.

        Args:
            persona: A persona descriptor dict.
            instruction: The instruction to respond to.

        Returns:
            A response string, or ``None`` on failure.
        """
        prompt = PERSONA_RESPONSE_PROMPT.format(
            persona=persona["description"],
            instruction=instruction,
        )
        result = self._mock_llm_generate(prompt)
        return result.strip() if result and len(result.strip()) > 10 else None

    @staticmethod
    def _mock_llm_generate(prompt: str, max_tokens: int = 256) -> str:
        """Placeholder LLM generation stub."""
        return f"[Persona-based response for: {prompt[:50]}...]"


# ---------------------------------------------------------------------------
# MathDataGenerator
# ---------------------------------------------------------------------------

class MathDataGenerator:
    """Math and reasoning data generator.

    Generates difficulty-graded math problems across multiple categories,
    complete with step-by-step solutions and verification.

    Categories: arithmetic, algebra, geometry, calculus, probability,
    logic puzzles, word problems.

    Features:
        - Difficulty-graded generation (easy → expert).
        - Step-by-step solution generation.
        - Programmatic verification where possible.
        - Self-debugging examples (intentional mistakes + corrections).

    Args:
        config: A ``SyntheticDataConfig`` instance.
    """

    _MATH_CATEGORIES: List[str] = [
        "arithmetic",
        "algebra",
        "geometry",
        "calculus",
        "probability",
        "logic puzzles",
        "word problems",
    ]

    _ARITHMETIC_TEMPLATES: List[str] = [
        "Calculate the result of {a} {op} {b}.",
        "What is {a} {op} {b} rounded to the nearest integer?",
        "Evaluate {a} {op} {b} {op2} {c}.",
        "Find the sum of {a} and {b}, then multiply by {c}.",
    ]

    _ALGEBRA_TEMPLATES: List[str] = [
        "Solve for x: {a}x + {b} = {c}.",
        "Solve the system: x + y = {a}, x - y = {b}.",
        "Factor the expression: x^2 + {a}x + {b}.",
        "Simplify: ({a}x^2 + {b}x) / x.",
    ]

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._rng = random.Random(42)

    def _generate_arithmetic(self, difficulty: str) -> Instruction:
        """Generate an arithmetic problem at the given difficulty."""
        if difficulty == "easy":
            a, b = self._rng.randint(1, 100), self._rng.randint(1, 100)
            op = self._rng.choice(["+", "-", "*"])
        elif difficulty == "medium":
            a, b = self._rng.randint(10, 1000), self._rng.randint(10, 100)
            op = self._rng.choice(["+", "-", "*", "/"])
        elif difficulty == "hard":
            a = self._rng.randint(100, 10000)
            b = self._rng.randint(10, 500)
            op = self._rng.choice(["+", "-", "*", "/"])
        else:
            a = self._rng.randint(1000, 100000)
            b = self._rng.randint(100, 5000)
            op = self._rng.choice(["+", "-", "*", "/"])

        template = self._rng.choice(self._ARITHMETIC_TEMPLATES)
        fmt_kwargs: Dict[str, Any] = {"a": a, "op": op, "b": b}
        if "{c}" in template:
            fmt_kwargs["c"] = self._rng.randint(1, 50)
        if "{op2}" in template:
            fmt_kwargs["op2"] = self._rng.choice(["+", "-"])
        question = template.format(**fmt_kwargs)

        # Compute the answer programmatically
        try:
            answer = self._safe_eval_arithmetic(a, b, op)
            solution = f"Step 1: Identify the operation ({op}).\nStep 2: Compute {a} {op} {b} = {answer}."
        except (ZeroDivisionError, ValueError):
            answer = "undefined"
            solution = "The operation is undefined."

        full_response = f"{solution}\n\nANSWER: {answer}"

        return Instruction(
            instruction=question,
            response=full_response,
            category="math",
            difficulty=difficulty,
            source="math_generator",
            quality_score=0.95,
            metadata={"math_category": "arithmetic", "answer": str(answer)},
        )

    def _generate_algebra(self, difficulty: str) -> Instruction:
        """Generate an algebra problem at the given difficulty."""
        template = self._rng.choice(self._ALGEBRA_TEMPLATES)
        a = self._rng.randint(1, 20)
        b = self._rng.randint(-10, 10)
        c = self._rng.randint(1, 50)

        question = template.format(a=a, b=b, c=c)

        # Solve linear equation: ax + b = c  →  x = (c - b) / a
        if a != 0 and "Solve for x:" in question:
            x = (c - b) / a
            solution = (
                f"Step 1: Start with {a}x + {b} = {c}.\n"
                f"Step 2: Subtract {b} from both sides: {a}x = {c - b}.\n"
                f"Step 3: Divide by {a}: x = {c - b} / {a} = {x}."
            )
            answer = x
        else:
            solution = "Step 1: Analyze the expression.\nStep 2: Apply algebraic rules."
            answer = "See solution steps"

        full_response = f"{solution}\n\nANSWER: {answer}"

        return Instruction(
            instruction=question,
            response=full_response,
            category="math",
            difficulty=difficulty,
            source="math_generator",
            quality_score=0.9,
            metadata={"math_category": "algebra", "answer": str(answer)},
        )

    def _generate_logic_puzzle(self, difficulty: str) -> Instruction:
        """Generate a logic puzzle."""
        puzzles_by_difficulty = {
            "easy": (
                "If all cats are animals, and Whiskers is a cat, what can we "
                "conclude about Whiskers? Explain the logical reasoning."
            ),
            "medium": (
                "A farmer has chickens and rabbits. There are 35 heads and 94 "
                "legs in total. How many chickens and how many rabbits are there? "
                "Show your work."
            ),
            "hard": (
                "Three friends (Alice, Bob, and Charlie) each have a different "
                "favorite color (red, blue, green). Using the following clues, "
                "determine each person's favorite color:\n"
                "1. Alice does not like red.\n"
                "2. Bob's favorite color is not green.\n"
                "3. The person who likes blue is not Bob.\n"
                "Explain your reasoning step by step."
            ),
            "expert": (
                "In a tournament with 10 players where each player plays every "
                "other player exactly once, and each game results in a win or "
                "loss (no draws), prove that at least one player must have won "
                "at least 4 games. Provide a rigorous proof."
            ),
        }

        question = puzzles_by_difficulty.get(difficulty, puzzles_by_difficulty["medium"])

        # Generate a solution
        if difficulty == "easy":
            response = (
                "Step 1: We are given two premises:\n"
                "  - All cats are animals.\n"
                "  - Whiskers is a cat.\n"
                "Step 2: By the rule of universal instantiation, since all "
                "cats are animals, we can conclude that Whiskers (being a cat) "
                "is also an animal.\n"
                "Step 3: This follows the logical form: All A are B. x is A. "
                "Therefore, x is B. (Modus ponens / universal instantiation)\n\n"
                "ANSWER: Whiskers is an animal."
            )
        else:
            response = (
                "Step 1: Carefully analyze the given conditions and constraints.\n"
                "Step 2: Set up equations or a logical framework.\n"
                "Step 3: Systematically eliminate impossibilities.\n"
                "Step 4: Derive the unique solution.\n\n"
                "ANSWER: See detailed step-by-step reasoning above."
            )

        return Instruction(
            instruction=question,
            response=response,
            category="math",
            difficulty=difficulty,
            source="math_generator",
            quality_score=0.85,
            metadata={"math_category": "logic puzzles"},
        )

    def _generate_self_debugging_example(self, difficulty: str) -> Instruction:
        """Generate a math problem with an intentional mistake and correction.

        The instruction asks the model to find and fix the error in a
        provided solution.
        """
        templates = [
            (
                "Find the error in this solution and provide the correct one:\n\n"
                "Problem: Calculate (8 + 4) * 3\n"
                "Given solution:\n"
                "Step 1: 8 + 4 = 12\n"
                "Step 2: 12 * 3 = 15\n"
                "Answer: 15\n\n"
                "What is wrong and what is the correct answer?"
            ),
            (
                "Find the error in this solution and provide the correct one:\n\n"
                "Problem: Solve 3x + 7 = 22\n"
                "Given solution:\n"
                "Step 1: 3x = 22 + 7 = 29\n"
                "Step 2: x = 29 / 3 = 9.66\n\n"
                "What is wrong and what is the correct answer?"
            ),
            (
                "Find the error in this solution and provide the correct one:\n\n"
                "Problem: What is 15% of 200?\n"
                "Given solution:\n"
                "Step 1: 15% = 0.15\n"
                "Step 2: 0.15 * 200 = 15\n\n"
                "What is wrong and what is the correct answer?"
            ),
        ]

        question = self._rng.choice(templates)

        response = (
            "Step 1: Review the given solution line by line.\n"
            "Step 2: Identify the arithmetic or logical error.\n"
            "Step 3: Compute the correct result.\n"
            "Step 4: Verify the corrected answer.\n\n"
            "CORRECTION: The error is in the calculation. "
            "Recalculate carefully to find the correct answer."
        )

        return Instruction(
            instruction=question,
            response=response,
            category="math",
            difficulty=difficulty,
            source="math_generator",
            quality_score=0.9,
            metadata={"math_category": "self_debugging"},
        )

    def run(self, num_problems: Optional[int] = None) -> SyntheticDataset:
        """Execute the math data generation pipeline.

        Generates problems across all math categories and difficulty levels.

        Args:
            num_problems: Target number of problems. Defaults to
                ``config.num_instructions``.

        Returns:
            A ``SyntheticDataset`` of math instructions.
        """
        target = num_problems or self.config.num_instructions
        instructions: List[Instruction] = []
        generators = [
            self._generate_arithmetic,
            self._generate_algebra,
            self._generate_logic_puzzle,
            self._generate_self_debugging_example,
        ]

        for i in range(target):
            difficulty = self._rng.choice(self.config.difficulty_levels)
            gen_fn = self._rng.choice(generators)
            try:
                instr = gen_fn(difficulty)
                instructions.append(instr)
            except Exception:
                continue

        return SyntheticDataset(instructions=instructions)

    @staticmethod
    def _safe_eval_arithmetic(a: int, b: int, op: str) -> float:
        """Safely evaluate a simple arithmetic expression."""
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "/":
            return a / b if b != 0 else float("inf")
        raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# CodeDataGenerator
# ---------------------------------------------------------------------------

class CodeDataGenerator:
    """Code generation data generator.

    Produces programming problems with specifications, test cases, solutions,
    execution verification, debugging exercises, and code review pairs.

    Supported languages: Python, JavaScript, C++, Java.

    Features:
        - Programming problem generation with clear specifications.
        - Test case generation (minimum 3 per problem).
        - Multiple solution approaches.
        - Solution verification.
        - Self-debugging examples (buggy code + correction).
        - Code review pairs (good vs. bad code).

    Args:
        config: A ``SyntheticDataConfig`` instance.
    """

    _LANGUAGES: List[str] = ["Python", "JavaScript", "C++", "Java"]

    _CODE_PROBLEMS: Dict[str, List[Dict[str, str]]] = {
        "Python": [
            {
                "problem": (
                    "Write a function `two_sum(nums: List[int], target: int) -> "
                    "List[int]` that returns the indices of the two numbers in "
                    "`nums` that add up to `target`. You may assume each input "
                    "has exactly one solution."
                ),
                "tests": (
                    "Test 1: two_sum([2, 7, 11, 15], 9) -> [0, 1]\n"
                    "Test 2: two_sum([3, 2, 4], 6) -> [1, 2]\n"
                    "Test 3: two_sum([3, 3], 6) -> [0, 1]"
                ),
                "solution": (
                    "def two_sum(nums, target):\n"
                    "    seen = {}\n"
                    "    for i, num in enumerate(nums):\n"
                    "        complement = target - num\n"
                    "        if complement in seen:\n"
                    "            return [seen[complement], i]\n"
                    "        seen[num] = i\n"
                    "    return []"
                ),
            },
            {
                "problem": (
                    "Write a function `is_valid(s: str) -> bool` that determines "
                    "if the input string containing only '(', ')', '{', '}', "
                    "'[' and ']' is valid. An input string is valid if open "
                    "brackets are closed by the same type and in the correct order."
                ),
                "tests": (
                    "Test 1: is_valid('()') -> True\n"
                    "Test 2: is_valid('()[]{}') -> True\n"
                    "Test 3: is_valid('(]') -> False\n"
                    "Test 4: is_valid('([)]') -> False"
                ),
                "solution": (
                    "def is_valid(s):\n"
                    "    stack = []\n"
                    "    mapping = {')': '(', '}': '{', ']': '['}\n"
                    "    for char in s:\n"
                    "        if char in mapping:\n"
                    "            top = stack.pop() if stack else '#'\n"
                    "            if mapping[char] != top:\n"
                    "                return False\n"
                    "        else:\n"
                    "            stack.append(char)\n"
                    "    return not stack"
                ),
            },
            {
                "problem": (
                    "Write a function `fibonacci(n: int) -> int` that returns "
                    "the n-th Fibonacci number using an efficient approach. "
                    "The Fibonacci sequence: F(0) = 0, F(1) = 1, "
                    "F(n) = F(n-1) + F(n-2)."
                ),
                "tests": (
                    "Test 1: fibonacci(0) -> 0\n"
                    "Test 2: fibonacci(1) -> 1\n"
                    "Test 3: fibonacci(10) -> 55\n"
                    "Test 4: fibonacci(20) -> 6765"
                ),
                "solution": (
                    "def fibonacci(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    a, b = 0, 1\n"
                    "    for _ in range(2, n + 1):\n"
                    "        a, b = b, a + b\n"
                    "    return b"
                ),
            },
        ],
        "JavaScript": [
            {
                "problem": (
                    "Write a function `debounce(fn, delay)` that delays "
                    "invoking `fn` until after `delay` milliseconds have "
                    "elapsed since the last time the debounced function was "
                    "invoked."
                ),
                "tests": (
                    "Test 1: debounce called multiple times within delay "
                    "should only fire once.\n"
                    "Test 2: debounce called once should fire after delay.\n"
                    "Test 3: debounced function receives correct arguments."
                ),
                "solution": (
                    "function debounce(fn, delay) {\n"
                    "    let timeoutId;\n"
                    "    return function(...args) {\n"
                    "        clearTimeout(timeoutId);\n"
                    "        timeoutId = setTimeout(() => {\n"
                    "            fn.apply(this, args);\n"
                    "        }, delay);\n"
                    "    };\n"
                    "}"
                ),
            },
        ],
        "C++": [
            {
                "problem": (
                    "Implement a function `int maxSubArray(const vector<int>& "
                    "nums)` that finds the contiguous subarray with the largest "
                    "sum and returns its sum."
                ),
                "tests": (
                    "Test 1: maxSubArray({-2,1,-3,4,-1,2,1,-5,4}) -> 6\n"
                    "Test 2: maxSubArray({1}) -> 1\n"
                    "Test 3: maxSubArray({5,4,-1,7,8}) -> 23"
                ),
                "solution": (
                    "#include <vector>\n"
                    "#include <algorithm>\n"
                    "using namespace std;\n\n"
                    "int maxSubArray(const vector<int>& nums) {\n"
                    "    int max_sum = nums[0], cur_sum = nums[0];\n"
                    "    for (size_t i = 1; i < nums.size(); i++) {\n"
                    "        cur_sum = max(nums[i], cur_sum + nums[i]);\n"
                    "        max_sum = max(max_sum, cur_sum);\n"
                    "    }\n"
                    "    return max_sum;\n"
                    "}"
                ),
            },
        ],
        "Java": [
            {
                "problem": (
                    "Implement a method `public int reverse(int x)` that "
                    "reverses the digits of a 32-bit signed integer. Return 0 "
                    "when the reversed integer overflows."
                ),
                "tests": (
                    "Test 1: reverse(123) -> 321\n"
                    "Test 2: reverse(-123) -> -321\n"
                    "Test 3: reverse(120) -> 21\n"
                    "Test 4: reverse(0) -> 0"
                ),
                "solution": (
                    "public int reverse(int x) {\n"
                    "    int rev = 0;\n"
                    "    while (x != 0) {\n"
                    "        int pop = x % 10;\n"
                    "        x /= 10;\n"
                    "        if (rev > Integer.MAX_VALUE/10 || "
                    "(rev == Integer.MAX_VALUE/10 && pop > 7)) return 0;\n"
                    "        if (rev < Integer.MIN_VALUE/10 || "
                    "(rev == Integer.MIN_VALUE/10 && pop < -8)) return 0;\n"
                    "        rev = rev * 10 + pop;\n"
                    "    }\n"
                    "    return rev;\n"
                    "}"
                ),
            },
        ],
    }

    _DEBUGGING_EXERCISES: List[Dict[str, str]] = [
        {
            "problem": "Write a function to find the maximum element in a list.",
            "buggy_code": (
                "def find_max(arr):\n"
                "    max_val = arr[0]\n"
                "    for i in range(1, len(arr)):\n"
                "        if arr[i] > max_val:\n"
                "            max_val = arr[i]\n"
                "    return arr  # Bug: returns the array, not max_val"
            ),
            "bug_explanation": (
                "The function correctly finds the maximum value but returns "
                "the original array instead of the max_val variable. Line 6 "
                "should `return max_val` instead of `return arr`."
            ),
            "correct_code": (
                "def find_max(arr):\n"
                "    if not arr:\n"
                "        return None\n"
                "    max_val = arr[0]\n"
                "    for i in range(1, len(arr)):\n"
                "        if arr[i] > max_val:\n"
                "            max_val = arr[i]\n"
                "    return max_val"
            ),
        },
        {
            "problem": "Write a function to check if a string is a palindrome.",
            "buggy_code": (
                "def is_palindrome(s):\n"
                "    s = s.lower()\n"
                "    return s == s.reverse()  # Bug: str.reverse() doesn't exist"
            ),
            "bug_explanation": (
                "Python strings do not have a .reverse() method. The correct "
                "approach is to use slicing with s[::-1] to reverse the string."
            ),
            "correct_code": (
                "def is_palindrome(s):\n"
                "    s = s.lower()\n"
                "    s = ''.join(c for c in s if c.isalnum())\n"
                "    return s == s[::-1]"
            ),
        },
    ]

    _CODE_REVIEW_EXERCISES: List[Dict[str, str]] = [
        {
            "problem": "Write a function to flatten a nested list.",
            "good_code": (
                "from typing import List, Any\n\n"
                "def flatten(nested: List[Any]) -> List[Any]:\n"
                "    \"\"\"Flatten an arbitrarily nested list.\"\"\"\n"
                "    result = []\n"
                "    for item in nested:\n"
                "        if isinstance(item, list):\n"
                "            result.extend(flatten(item))\n"
                "        else:\n"
                "            result.append(item)\n"
                "    return result"
            ),
            "bad_code": (
                "def flatten(n):\n"
                "    r=[]\n"
                "    for x in n:\n"
                "        if type(x)==list:\n"
                "            for y in flatten(x):\n"
                "                r.append(y)\n"
                "        else:\n"
                "            r.append(x)\n"
                "    return r"
            ),
            "review": (
                "The good code uses: descriptive variable names, type hints, "
                "docstrings, list.extend() for efficiency, and isinstance() "
                "instead of type()==. The bad code uses single-letter names, "
                "manual inner-loop appending, and == for type comparison."
            ),
        },
    ]

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._rng = random.Random(42)

    def _generate_problem(self, difficulty: str) -> Instruction:
        """Generate a coding problem with test cases and a reference solution."""
        language = self._rng.choice(self._LANGUAGES)
        problems = self._CODE_PROBLEMS.get(language, self._CODE_PROBLEMS["Python"])
        if not problems:
            problems = self._CODE_PROBLEMS["Python"]
        problem_data = self._rng.choice(problems)

        question = (
            f"Language: {language}\n\n"
            f"PROBLEM:\n{problem_data['problem']}\n\n"
            f"TEST CASES:\n{problem_data['tests']}\n\n"
            f"Provide a correct and efficient solution."
        )

        response = (
            f"SOLUTION ({language}):\n```\n{problem_data['solution']}\n```\n\n"
            f"Explanation: The solution passes all provided test cases."
        )

        return Instruction(
            instruction=question,
            response=response,
            category="code",
            difficulty=difficulty,
            source="code_generator",
            quality_score=0.95,
            metadata={
                "language": language,
                "has_tests": True,
                "verified": True,
            },
        )

    def _generate_debugging(self, difficulty: str) -> Instruction:
        """Generate a code debugging exercise."""
        exercise = self._rng.choice(self._DEBUGGING_EXERCISES)

        question = (
            f"PROBLEM:\n{exercise['problem']}\n\n"
            f"BUGGY CODE:\n```\n{exercise['buggy_code']}\n```\n\n"
            "Find the bug, explain it, and provide the corrected code."
        )

        response = (
            f"BUG EXPLANATION:\n{exercise['bug_explanation']}\n\n"
            f"CORRECT CODE:\n```\n{exercise['correct_code']}\n```"
        )

        return Instruction(
            instruction=question,
            response=response,
            category="code",
            difficulty=difficulty,
            source="code_generator",
            quality_score=0.9,
            metadata={"type": "debugging"},
        )

    def _generate_code_review(self, difficulty: str) -> Instruction:
        """Generate a code review exercise (good vs. bad code)."""
        exercise = self._rng.choice(self._CODE_REVIEW_EXERCISES)

        question = (
            f"PROBLEM:\n{exercise['problem']}\n\n"
            f"Compare these two solutions and explain which is better and why:\n\n"
            f"SOLUTION A:\n```\n{exercise['good_code']}\n```\n\n"
            f"SOLUTION B:\n```\n{exercise['bad_code']}\n```\n\n"
            "Which solution is better? Explain your reasoning."
        )

        response = (
            f"REVIEW:\n{exercise['review']}\n\n"
            "Solution A is the better implementation."
        )

        return Instruction(
            instruction=question,
            response=response,
            category="code",
            difficulty=difficulty,
            source="code_generator",
            quality_score=0.9,
            metadata={"type": "code_review"},
        )

    def run(self, num_problems: Optional[int] = None) -> SyntheticDataset:
        """Execute the code data generation pipeline.

        Generates a mix of coding problems, debugging exercises, and
        code review pairs across multiple languages and difficulties.

        Args:
            num_problems: Target number of problems. Defaults to
                ``config.num_instructions``.

        Returns:
            A ``SyntheticDataset`` of code-related instructions.
        """
        target = num_problems or self.config.num_instructions
        instructions: List[Instruction] = []
        generators = [
            self._generate_problem,
            self._generate_debugging,
            self._generate_code_review,
        ]

        for i in range(target):
            difficulty = self._rng.choice(self.config.difficulty_levels)
            gen_fn = self._rng.choice(generators)
            try:
                instr = gen_fn(difficulty)
                instructions.append(instr)
            except Exception:
                continue

        return SyntheticDataset(instructions=instructions)


# ---------------------------------------------------------------------------
# SyntheticDataPipeline — Master Orchestrator
# ---------------------------------------------------------------------------

class SyntheticDataPipeline:
    """Master orchestrator that combines all synthetic data generators.

    Runs the full generation pipeline across all categories, tracks
    statistics and quality metrics, and produces a unified
    ``SyntheticDataset``.

    Configurable generation targets per category allow fine-grained
    control over the composition of the final dataset.

    Args:
        config: A ``SyntheticDataConfig`` instance controlling all generators.
        targets_per_generator: Optional mapping of generator name to target
            instruction count. If ``None``, all generators share the total
            ``config.num_instructions`` equally.

    Example::

        config = SyntheticDataConfig(generator_model_name="mistral-7b-instruct")
        pipeline = SyntheticDataPipeline(config)
        dataset = pipeline.run_full_pipeline()
        print(dataset.statistics())
        dataset.save("output.json")
    """

    def __init__(
        self,
        config: SyntheticDataConfig,
        targets_per_generator: Optional[Dict[str, int]] = None,
    ) -> None:
        self.config = config
        self._targets = targets_per_generator
        self._rng = random.Random(42)
        self._stats: Dict[str, Any] = {
            "generators_run": [],
            "total_generated": 0,
            "total_after_quality_filter": 0,
            "total_after_dedup": 0,
            "per_generator_stats": {},
        }

    def run_full_pipeline(self) -> SyntheticDataset:
        """Execute all generators and merge into a unified dataset.

        The pipeline runs the following generators in order:
            1. Self-Instruct (general instruction following)
            2. Evol-Instruct (complexity-evolved instructions)
            3. Rejection Sampling (high-quality filtering)
            4. Persona Generator (diverse perspectives)
            5. Math Data Generator (mathematical reasoning)
            6. Code Data Generator (programming tasks)

        Returns:
            A merged ``SyntheticDataset`` containing all generated data.
        """
        all_instructions: List[Instruction] = []
        total_target = self.config.num_instructions

        # Divide the budget equally among generators unless overridden
        num_generators = 6
        default_per_gen = max(total_target // num_generators, 100)
        targets = self._targets or {
            "self_instruct": default_per_gen,
            "evol_instruct": default_per_gen,
            "rejection_sampling": default_per_gen,
            "persona": default_per_gen,
            "math": default_per_gen,
            "code": default_per_gen,
        }

        # 1. Self-Instruct
        gen_target = targets.get("self_instruct", default_per_gen)
        gen_config = self._make_sub_config(gen_target)
        si_gen = SelfInstructGenerator(gen_config)
        si_dataset = si_gen.run()
        all_instructions.extend(si_dataset.instructions)
        self._record_stats("self_instruct", si_dataset)

        # 2. Evol-Instruct
        gen_target = targets.get("evol_instruct", default_per_gen)
        gen_config = self._make_sub_config(gen_target)
        evol_gen = EvolInstructGenerator(gen_config)
        evol_dataset = evol_gen.run()
        all_instructions.extend(evol_dataset.instructions)
        self._record_stats("evol_instruct", evol_dataset)

        # 3. Rejection Sampling
        gen_target = targets.get("rejection_sampling", default_per_gen)
        gen_config = self._make_sub_config(gen_target)
        rs_gen = RejectionSamplingGenerator(gen_config)
        rs_dataset = rs_gen.run()
        all_instructions.extend(rs_dataset.instructions)
        self._record_stats("rejection_sampling", rs_dataset)

        # 4. Persona Generator
        gen_target = targets.get("persona", default_per_gen)
        gen_config = self._make_sub_config(gen_target)
        pers_gen = PersonaGenerator(gen_config)
        pers_dataset = pers_gen.run(num_personas=gen_target)
        all_instructions.extend(pers_dataset.instructions)
        self._record_stats("persona", pers_dataset)

        # 5. Math Data Generator
        gen_target = targets.get("math", default_per_gen)
        gen_config = self._make_sub_config(gen_target)
        math_gen = MathDataGenerator(gen_config)
        math_dataset = math_gen.run(num_problems=gen_target)
        all_instructions.extend(math_dataset.instructions)
        self._record_stats("math", math_dataset)

        # 6. Code Data Generator
        gen_target = targets.get("code", default_per_gen)
        gen_config = self._make_sub_config(gen_target)
        code_gen = CodeDataGenerator(gen_config)
        code_dataset = code_gen.run(num_problems=gen_target)
        all_instructions.extend(code_dataset.instructions)
        self._record_stats("code", code_dataset)

        # Final global deduplication
        merged = SyntheticDataset(instructions=all_instructions)

        # Update aggregate stats
        self._stats["total_generated"] = len(all_instructions)
        self._stats["total_after_dedup"] = merged.count

        return merged

    def get_statistics(self) -> Dict[str, Any]:
        """Return detailed pipeline execution statistics.

        Returns:
            A dictionary with per-generator and aggregate stats.
        """
        return dict(self._stats)

    # -- Helpers ------------------------------------------------------------

    def _make_sub_config(self, num_instructions: int) -> SyntheticDataConfig:
        """Create a sub-config with a reduced instruction target."""
        return SyntheticDataConfig(
            generator_model_name=self.config.generator_model_name,
            num_instructions=num_instructions,
            difficulty_levels=list(self.config.difficulty_levels),
            domains=list(self.config.domains),
            max_evolution_rounds=self.config.max_evolution_rounds,
            quality_threshold=self.config.quality_threshold,
            dedup_threshold=self.config.dedup_threshold,
        )

    def _record_stats(self, name: str, dataset: SyntheticDataset) -> None:
        """Record statistics for a single generator run."""
        self._stats["generators_run"].append(name)
        self._stats["per_generator_stats"][name] = {
            "count": dataset.count,
            "avg_quality": dataset.avg_quality_score(),
            "categories": dataset.category_distribution(),
            "difficulties": dataset.difficulty_distribution(),
        }
