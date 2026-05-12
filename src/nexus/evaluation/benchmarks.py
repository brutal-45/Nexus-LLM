"""
Benchmark Evaluation Harness
==============================
Run standardized benchmarks to evaluate model capabilities.

Supported benchmarks:
    - MMLU: Massive Multitask Language Understanding (57 subjects, 14K questions)
    - GSM8K: Grade School Math (8.5K math word problems)
    - HumanEval: Code generation (164 Python programming problems)
    - HellaSwag: Commonsense reasoning (10K sentence completion)
    - ARC: AI2 Reasoning Challenge (7.8K science questions)
    - TruthfulQA: Truthfulness (817 questions)
    - WinoGrande: Common sense reasoning (1.3K pronoun resolution)

For each benchmark:
    1. Load the dataset
    2. Format prompts appropriately
    3. Generate completions
    4. Score answers
    5. Report accuracy and per-category breakdown
"""

from __future__ import annotations
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..model.transformer import NexusTransformer
from ..data.tokenizer import BPETokenizer
from ..inference.generator import TextGenerator, GenerationConfig


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""
    benchmark_name: str
    accuracy: float
    total_questions: int
    correct: int
    per_category_accuracy: Dict[str, float] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)
    time_seconds: float = 0.0


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    max_new_tokens: int = 128
    temperature: float = 0.0  # Greedy for reproducibility
    top_p: float = 1.0
    batch_size: int = 8
    num_few_shot: int = 0  # 0-shot by default


class BenchmarkHarness:
    """
    Main benchmark evaluation harness.
    
    Usage:
        harness = BenchmarkHarness(model, tokenizer)
        results = harness.run(["mmlu", "gsm8k", "humaneval"])
        harness.print_results(results)
    """

    # Benchmark registry
    BENCHMARKS = {
        "mmlu": {"name": "MMLU", "description": "Massive Multitask Language Understanding"},
        "gsm8k": {"name": "GSM8K", "description": "Grade School Math"},
        "humaneval": {"name": "HumanEval", "description": "Code Generation"},
        "hellaswag": {"name": "HellaSwag", "description": "Commonsense Reasoning"},
        "arc": {"name": "ARC", "description": "AI2 Reasoning Challenge"},
        "truthfulqa": {"name": "TruthfulQA", "description": "Truthfulness"},
    }

    def __init__(
        self,
        model: NexusTransformer,
        tokenizer: BPETokenizer,
        config: Optional[BenchmarkConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BenchmarkConfig()
        self.generator = TextGenerator(model, tokenizer)

    def run(self, benchmarks: List[str]) -> Dict[str, BenchmarkResult]:
        """
        Run specified benchmarks.
        
        Args:
            benchmarks: List of benchmark names to run.
        
        Returns:
            Dictionary mapping benchmark name to results.
        """
        results = {}
        
        for bench_name in benchmarks:
            bench_lower = bench_name.lower()
            
            if bench_lower == "mmlu":
                results[bench_name] = self._run_mmlu()
            elif bench_lower == "gsm8k":
                results[bench_name] = self._run_gsm8k()
            elif bench_lower == "humaneval":
                results[bench_name] = self._run_humaneval()
            elif bench_lower == "hellaswag":
                results[bench_name] = self._run_hellaswag()
            elif bench_lower == "arc":
                results[bench_name] = self._run_arc()
            else:
                print(f"  [Warning] Unknown benchmark: {bench_name}, skipping")
        
        return results

    def _run_mmlu(self) -> BenchmarkResult:
        """Run MMLU benchmark (multiple choice, 57 subjects)."""
        print(f"\n{'='*50}")
        print(f"Running MMLU (Massive Multitask Language Understanding)")
        print(f"{'='*50}")
        
        start = time.time()
        
        # MMLU format: Multiple choice with 4 options (A/B/C/D)
        # For demo, we use synthetic MMLU-style questions
        # In production, load from datasets library
        
        questions = self._get_mmlu_questions()
        
        correct = 0
        total = len(questions)
        per_subject: Dict[str, Tuple[int, int]] = {}
        
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        for q in questions:
            prompt = self._format_mmlu_prompt(q, num_few_shot=self.config.num_few_shot)
            result = self.generator.generate(prompt, gen_config)
            answer = result.generated_text[0].strip().upper()
            
            predicted = self._extract_mmlu_answer(answer)
            correct_answer = q["answer"]
            
            if predicted == correct_answer:
                correct += 1
            
            subject = q.get("subject", "unknown")
            if subject not in per_subject:
                per_subject[subject] = [0, 0]
            per_subject[subject][1] += 1
            if predicted == correct_answer:
                per_subject[subject][0] += 1
        
        elapsed = time.time() - start
        accuracy = correct / total if total > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name="MMLU",
            accuracy=accuracy,
            total_questions=total,
            correct=correct,
            per_category_accuracy={
                k: v[0] / v[1] if v[1] > 0 else 0.0
                for k, v in per_subject.items()
            },
            time_seconds=elapsed,
        )
        
        print(f"  MMLU Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Time: {elapsed:.1f}s")
        
        return result

    def _run_gsm8k(self) -> BenchmarkResult:
        """Run GSM8K benchmark (math word problems)."""
        print(f"\n{'='*50}")
        print(f"Running GSM8K (Grade School Math)")
        print(f"{'='*50}")
        
        start = time.time()
        questions = self._get_gsm8k_questions()
        
        correct = 0
        total = len(questions)
        
        gen_config = GenerationConfig(
            max_new_tokens=256,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        for q in questions:
            prompt = self._format_gsm8k_prompt(q["question"])
            result = self.generator.generate(prompt, gen_config)
            answer = result.generated_text[0].strip()
            
            # Extract numeric answer
            predicted = self._extract_numeric_answer(answer)
            gold = self._extract_numeric_answer(q["answer"])
            
            if predicted is not None and gold is not None and predicted == gold:
                correct += 1
        
        elapsed = time.time() - start
        accuracy = correct / total if total > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name="GSM8K",
            accuracy=accuracy,
            total_questions=total,
            correct=correct,
            time_seconds=elapsed,
        )
        
        print(f"  GSM8K Accuracy: {accuracy:.2%} ({correct}/{total})")
        return result

    def _run_humaneval(self) -> BenchmarkResult:
        """Run HumanEval benchmark (code generation)."""
        print(f"\n{'='*50}")
        print(f"Running HumanEval (Code Generation)")
        print(f"{'='*50}")
        
        start = time.time()
        problems = self._get_humaneval_problems()
        
        correct = 0
        total = len(problems)
        
        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.2,  # Slightly random for code
            top_p=0.95,
        )
        
        for problem in problems:
            prompt = problem["prompt"]
            result = self.generator.generate(prompt, gen_config)
            generated_code = prompt + result.generated_text[0]
            
            # Execute tests
            is_correct = self._check_humaneval_solution(
                generated_code, problem["test"]
            )
            if is_correct:
                correct += 1
        
        elapsed = time.time() - start
        accuracy = correct / total if total > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name="HumanEval",
            accuracy=accuracy,
            total_questions=total,
            correct=correct,
            time_seconds=elapsed,
        )
        
        print(f"  HumanEval pass@1: {accuracy:.2%} ({correct}/{total})")
        return result

    def _run_hellaswag(self) -> BenchmarkResult:
        """Run HellaSwag benchmark (commonsense sentence completion)."""
        print(f"\n{'='*50}")
        print(f"Running HellaSwag (Commonsense Reasoning)")
        print(f"{'='*50}")
        
        start = time.time()
        questions = self._get_hellaswag_questions()
        
        correct = 0
        total = len(questions)
        
        gen_config = GenerationConfig(
            max_new_tokens=32,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        for q in questions:
            prompt = q["context"]
            result = self.generator.generate(prompt, gen_config)
            prediction = result.generated_text[0].strip().lower()
            
            # Check if prediction matches the correct ending
            gold = q["endings"][q["answer"]].lower()
            if gold.startswith(prediction[:50]):
                correct += 1
        
        elapsed = time.time() - start
        accuracy = correct / total if total > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name="HellaSwag",
            accuracy=accuracy,
            total_questions=total,
            correct=correct,
            time_seconds=elapsed,
        )
        
        print(f"  HellaSwag Accuracy: {accuracy:.2%} ({correct}/{total})")
        return result

    def _run_arc(self) -> BenchmarkResult:
        """Run ARC benchmark (science reasoning)."""
        print(f"\n{'='*50}")
        print(f"Running ARC (AI2 Reasoning Challenge)")
        print(f"{'='*50}")
        
        start = time.time()
        questions = self._get_arc_questions()
        
        correct = 0
        total = len(questions)
        
        gen_config = GenerationConfig(
            max_new_tokens=16,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        for q in questions:
            prompt = self._format_arc_prompt(q)
            result = self.generator.generate(prompt, gen_config)
            answer = result.generated_text[0].strip().upper()
            predicted = self._extract_mmlu_answer(answer)
            
            if predicted == q["answer"]:
                correct += 1
        
        elapsed = time.time() - start
        accuracy = correct / total if total > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name="ARC",
            accuracy=accuracy,
            total_questions=total,
            correct=correct,
            time_seconds=elapsed,
        )
        
        print(f"  ARC Accuracy: {accuracy:.2%} ({correct}/{total})")
        return result

    # === Prompt Formatting ===

    def _format_mmlu_prompt(self, question: Dict, num_few_shot: int = 0) -> str:
        """Format MMLU multiple-choice question."""
        prompt = f"{question['question']}\n"
        for label, text in zip(["A", "B", "C", "D"], question["choices"]):
            prompt += f"{label}. {text}\n"
        prompt += "Answer: The answer is"
        return prompt

    def _format_gsm8k_prompt(self, question: str) -> str:
        """Format GSM8K math problem."""
        return f"Question: {question}\n\nSolution: Let's think step by step.\n"

    def _format_arc_prompt(self, question: Dict) -> str:
        """Format ARC question."""
        prompt = f"{question['question']}\n"
        for label, text in zip(["A", "B", "C", "D"], question["choices"]):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        return prompt

    @staticmethod
    def _extract_mmlu_answer(text: str) -> str:
        """Extract A/B/C/D from model output."""
        text = text.strip().upper()
        if text and text[0] in "ABCD":
            return text[0]
        return ""

    @staticmethod
    def _extract_numeric_answer(text: str) -> Optional[int]:
        """Extract a numeric answer from text."""
        # Look for "#### 42" format (GSM8K) or just last number
        match = re.search(r'####\s*(-?\d+)', text)
        if match:
            return int(match.group(1))
        match = re.search(r'=\s*(-?\d+)', text)
        if match:
            return int(match.group(1))
        numbers = re.findall(r'-?\d+', text)
        return int(numbers[-1]) if numbers else None

    @staticmethod
    def _check_humaneval_solution(code: str, test_code: str) -> bool:
        """Check if generated code passes the tests."""
        try:
            exec_globals = {}
            exec(code + "\n" + test_code, exec_globals)
            result = exec_globals.get("check", lambda: False)()
            return bool(result)
        except Exception:
            return False

    # === Sample Questions (for demo/testing) ===

    @staticmethod
    def _get_mmlu_questions() -> List[Dict]:
        """Get sample MMLU questions (replace with actual dataset)."""
        return [
            {
                "subject": "math",
                "question": "What is the derivative of x^3 with respect to x?",
                "choices": ["x^2", "3x^2", "3x", "x^3/3"],
                "answer": "B",
            },
            {
                "subject": "physics",
                "question": "What is the speed of light in vacuum (approximately)?",
                "choices": ["3 x 10^6 m/s", "3 x 10^8 m/s", "3 x 10^10 m/s", "3 x 10^12 m/s"],
                "answer": "B",
            },
        ]

    @staticmethod
    def _get_gsm8k_questions() -> List[Dict]:
        return [
            {
                "question": "If Alice has 5 apples and gives 2 to Bob, how many does she have left?",
                "answer": "#### 3",
            },
        ]

    @staticmethod
    def _get_humaneval_problems() -> List[Dict]:
        return [
            {
                "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
                "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\ncheck = lambda: True",
            },
        ]

    @staticmethod
    def _get_hellaswag_questions() -> List[Dict]:
        return [
            {
                "context": "The man picked up the heavy box and",
                "endings": ["dropped it on his foot.", "put it down gently.", "threw it at the wall.", "sat on it."],
                "answer": 1,
            },
        ]

    @staticmethod
    def _get_arc_questions() -> List[Dict]:
        return [
            {
                "question": "What is the chemical symbol for water?",
                "choices": ["H2O", "CO2", "NaCl", "O2"],
                "answer": "A",
            },
        ]

    # === Reporting ===

    @staticmethod
    def print_results(results: Dict[str, BenchmarkResult]):
        """Print all benchmark results in a table."""
        print(f"\n{'='*70}")
        print(f"{'Benchmark Results':^70}")
        print(f"{'='*70}")
        print(f"{'Benchmark':<20} {'Accuracy':<12} {'Correct':<12} {'Total':<10} {'Time':<10}")
        print(f"{'-'*70}")
        
        for name, result in results.items():
            print(f"{name:<20} {result.accuracy:>10.2%} {result.correct:>10} {result.total_questions:>8} {result.time_seconds:>8.1f}s")
        
        print(f"{'='*70}")
