"""
Benchmark Runner Module

Runs simulated standard NLP benchmarks: MMLU, HellaSwag, ARC, WinoGrande.
Each benchmark produces accuracy-based scores with category breakdowns.
"""

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Supported benchmark types."""
    MMLU = "mmlu"
    HELLASWAG = "hellaswag"
    ARC = "arc"
    WINOGRANDE = "winogrande"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    benchmark_type: BenchmarkType
    num_examples: int = 100
    seed: int = 42
    few_shot: int = 0
    batch_size: int = 32
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_type": self.benchmark_type.value,
            "num_examples": self.num_examples,
            "seed": self.seed,
            "few_shot": self.few_shot,
            "batch_size": self.batch_size,
            "categories": self.categories,
        }


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    benchmark_type: BenchmarkType
    model_name: str
    overall_score: float
    category_scores: Dict[str, float] = field(default_factory=dict)
    num_examples: int = 0
    num_correct: int = 0
    elapsed_seconds: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    per_example: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_type": self.benchmark_type.value,
            "model_name": self.model_name,
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "elapsed_seconds": self.elapsed_seconds,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BenchmarkResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["benchmark_type"] = BenchmarkType(data["benchmark_type"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Simulated benchmark data generators
# ---------------------------------------------------------------------------

_MMLU_CATEGORIES = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

_ARC_CATEGORIES = [
    "easy", "medium", "hard",
]

_WINOGRANDE_CATEGORIES = [
    "coreference_resolution",
]


def _generate_mmlu_example(rng: random.Random, category: str) -> Dict[str, Any]:
    """Generate a simulated MMLU multiple-choice question."""
    templates = [
        {
            "question": f"Which of the following best describes a key concept in {category}?",
            "choices": ["Option A", "Option B", "Option C", "Option D"],
            "answer": rng.choice([0, 1, 2, 3]),
        },
        {
            "question": f"In the field of {category}, what is the primary purpose of the standard methodology?",
            "choices": ["To establish norms", "To reduce complexity", "To ensure validity", "All of the above"],
            "answer": rng.choice([0, 1, 2, 3]),
        },
        {
            "question": f"Which statement about {category} is most accurate?",
            "choices": [
                "It has no practical applications",
                "It is purely theoretical",
                "It integrates theory and practice",
                "It is obsolete",
            ],
            "answer": 2,
        },
    ]
    template = rng.choice(templates)
    return {
        "category": category,
        "question": template["question"],
        "choices": template["choices"],
        "answer": template["answer"],
    }


def _generate_hellaswag_example(rng: random.Random) -> Dict[str, Any]:
    """Generate a simulated HellaSwag sentence completion example."""
    contexts = [
        "A person is walking down the street and",
        "The chef prepares the meal by first",
        "The student studies for the exam by",
        "The scientist conducts the experiment by",
        "The athlete trains for the competition by",
    ]
    endings_sets = [
        ["enters a building.", "crosses the road.", "waves at a friend.", "drops their keys."],
        ["chopping vegetables.", "turning on the oven.", "setting the table.", "reading a recipe."],
        ["reviewing notes.", "watching TV.", "sleeping early.", "calling a friend."],
        ["setting up equipment.", "writing a paper.", "grading exams.", "attending a meeting."],
        ["running laps.", "reading a book.", "cooking dinner.", "painting a picture."],
    ]
    idx = rng.randint(0, len(contexts) - 1)
    return {
        "context": contexts[idx],
        "endings": endings_sets[idx],
        "answer": rng.randint(0, 3),
    }


def _generate_arc_example(rng: random.Random, difficulty: str) -> Dict[str, Any]:
    """Generate a simulated ARC science question."""
    questions = {
        "easy": [
            {"question": "What is the boiling point of water at sea level?", "choices": ["50°C", "75°C", "100°C", "125°C"], "answer": 2},
            {"question": "Which planet is closest to the Sun?", "choices": ["Venus", "Mercury", "Earth", "Mars"], "answer": 1},
            {"question": "What gas do plants absorb from the atmosphere?", "choices": ["Oxygen", "Nitrogen", "Carbon Dioxide", "Hydrogen"], "answer": 2},
        ],
        "medium": [
            {"question": "What type of rock is formed from cooled magma?", "choices": ["Sedimentary", "Metamorphic", "Igneous", "Limestone"], "answer": 2},
            {"question": "Which organelle is responsible for energy production in cells?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi body"], "answer": 2},
            {"question": "What is the chemical formula for table salt?", "choices": ["NaO", "NaCl", "KCl", "CaCl2"], "answer": 1},
        ],
        "hard": [
            {"question": "What is the primary driver of tectonic plate movement?", "choices": ["Gravitational pull", "Convection currents", "Magnetic forces", "Solar radiation"], "answer": 1},
            {"question": "In quantum mechanics, what principle states that certain pairs of properties cannot both be precisely known?", "choices": ["Pauli exclusion", "Superposition", "Uncertainty principle", "Complementarity"], "answer": 2},
            {"question": "Which enzyme unwinds the DNA double helix during replication?", "choices": ["Polymerase", "Ligase", "Helicase", "Primase"], "answer": 2},
        ],
    }
    return rng.choice(questions.get(difficulty, questions["medium"]))


def _generate_winogrande_example(rng: random.Random) -> Dict[str, Any]:
    """Generate a simulated WinoGrande coreference resolution example."""
    examples = [
        {"sentence": "The trophy didn't fit into the brown suitcase because ___ was too small.", "choices": ["the trophy", "the suitcase"], "answer": 1},
        {"sentence": "The city councilmen refused the demonstrators a permit because ___ feared violence.", "choices": ["the councilmen", "the demonstrators"], "answer": 0},
        {"sentence": "Joan made sure to thank Susan for all the help she had ___.", "choices": ["given", "received"], "answer": 1},
        {"sentence": "The police arrested the protesters because ___ threatened public safety.", "choices": ["the police", "the protesters"], "answer": 1},
        {"sentence": "Mary gave Joan a beautiful bouquet because ___ was graduating.", "choices": ["Mary", "Joan"], "answer": 1},
    ]
    return rng.choice(examples)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs simulated NLP benchmarks.

    Each benchmark generates synthetic questions, simulates model predictions
    based on a configurable accuracy profile, and reports category-level and
    overall scores.

    The simulated accuracy is controlled via ``model_accuracy`` (global baseline)
    and ``category_accuracy_offset`` (per-category adjustments).  This allows
    realistic-looking results without requiring an actual model inference backend.
    """

    def __init__(
        self,
        model_name: str = "simulated-model",
        model_accuracy: float = 0.65,
        category_accuracy_offset: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.model_accuracy = max(0.0, min(1.0, model_accuracy))
        self.category_accuracy_offset = category_accuracy_offset or {}
        self.seed = seed

    def _effective_accuracy(self, category: str) -> float:
        offset = self.category_accuracy_offset.get(category, 0.0)
        return max(0.0, min(1.0, self.model_accuracy + offset))

    def _simulate_prediction(self, rng: random.Random, answer: int, num_choices: int, accuracy: float) -> int:
        """Simulate a model prediction given ground truth and expected accuracy."""
        if rng.random() < accuracy:
            return answer
        wrong = list(range(num_choices))
        wrong.remove(answer)
        return rng.choice(wrong)

    # ----- Individual benchmark runners -----

    def run_mmlu(self, config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
        """Run simulated MMLU benchmark."""
        if config is None:
            config = BenchmarkConfig(benchmark_type=BenchmarkType.MMLU)
        rng = random.Random(config.seed)
        categories = config.categories or rng.sample(_MMLU_CATEGORIES, min(10, len(_MMLU_CATEGORIES)))
        examples_per_cat = max(1, config.num_examples // len(categories))

        start = time.time()
        category_scores: Dict[str, float] = {}
        total_correct = 0
        total_examples = 0
        per_example: List[Dict[str, Any]] = []

        for cat in categories:
            correct = 0
            for _ in range(examples_per_cat):
                ex = _generate_mmlu_example(rng, cat)
                acc = self._effective_accuracy(cat)
                pred = self._simulate_prediction(rng, ex["answer"], len(ex["choices"]), acc)
                is_correct = pred == ex["answer"]
                correct += int(is_correct)
                per_example.append({
                    "category": cat,
                    "question": ex["question"],
                    "predicted": pred,
                    "ground_truth": ex["answer"],
                    "correct": is_correct,
                })
            cat_score = correct / examples_per_cat
            category_scores[cat] = round(cat_score, 4)
            total_correct += correct
            total_examples += examples_per_cat

        overall = total_correct / total_examples if total_examples > 0 else 0.0
        return BenchmarkResult(
            benchmark_type=BenchmarkType.MMLU,
            model_name=self.model_name,
            overall_score=round(overall, 4),
            category_scores=category_scores,
            num_examples=total_examples,
            num_correct=total_correct,
            elapsed_seconds=time.time() - start,
            config=config.to_dict(),
            per_example=per_example,
        )

    def run_hellaswag(self, config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
        """Run simulated HellaSwag benchmark."""
        if config is None:
            config = BenchmarkConfig(benchmark_type=BenchmarkType.HELLASWAG)
        rng = random.Random(config.seed)

        start = time.time()
        correct = 0
        per_example: List[Dict[str, Any]] = []

        for i in range(config.num_examples):
            ex = _generate_hellaswag_example(rng)
            acc = self._effective_accuracy("hellaswag")
            pred = self._simulate_prediction(rng, ex["answer"], len(ex["endings"]), acc)
            is_correct = pred == ex["answer"]
            correct += int(is_correct)
            per_example.append({
                "context": ex["context"],
                "predicted": pred,
                "ground_truth": ex["answer"],
                "correct": is_correct,
            })

        overall = correct / config.num_examples if config.num_examples > 0 else 0.0
        return BenchmarkResult(
            benchmark_type=BenchmarkType.HELLASWAG,
            model_name=self.model_name,
            overall_score=round(overall, 4),
            category_scores={"overall": round(overall, 4)},
            num_examples=config.num_examples,
            num_correct=correct,
            elapsed_seconds=time.time() - start,
            config=config.to_dict(),
            per_example=per_example,
        )

    def run_arc(self, config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
        """Run simulated ARC benchmark."""
        if config is None:
            config = BenchmarkConfig(benchmark_type=BenchmarkType.ARC)
        rng = random.Random(config.seed)
        difficulties = config.categories or _ARC_CATEGORIES
        examples_per_diff = max(1, config.num_examples // len(difficulties))

        start = time.time()
        category_scores: Dict[str, float] = {}
        total_correct = 0
        total_examples = 0
        per_example: List[Dict[str, Any]] = []

        for diff in difficulties:
            correct = 0
            for _ in range(examples_per_diff):
                ex = _generate_arc_example(rng, diff)
                acc = self._effective_accuracy(f"arc_{diff}")
                pred = self._simulate_prediction(rng, ex["answer"], len(ex["choices"]), acc)
                is_correct = pred == ex["answer"]
                correct += int(is_correct)
                per_example.append({
                    "difficulty": diff,
                    "question": ex["question"],
                    "predicted": pred,
                    "ground_truth": ex["answer"],
                    "correct": is_correct,
                })
            cat_score = correct / examples_per_diff
            category_scores[diff] = round(cat_score, 4)
            total_correct += correct
            total_examples += examples_per_diff

        overall = total_correct / total_examples if total_examples > 0 else 0.0
        return BenchmarkResult(
            benchmark_type=BenchmarkType.ARC,
            model_name=self.model_name,
            overall_score=round(overall, 4),
            category_scores=category_scores,
            num_examples=total_examples,
            num_correct=total_correct,
            elapsed_seconds=time.time() - start,
            config=config.to_dict(),
            per_example=per_example,
        )

    def run_winogrande(self, config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
        """Run simulated WinoGrande benchmark."""
        if config is None:
            config = BenchmarkConfig(benchmark_type=BenchmarkType.WINOGRANDE)
        rng = random.Random(config.seed)

        start = time.time()
        correct = 0
        per_example: List[Dict[str, Any]] = []

        for _ in range(config.num_examples):
            ex = _generate_winogrande_example(rng)
            acc = self._effective_accuracy("winogrande")
            pred = self._simulate_prediction(rng, ex["answer"], len(ex["choices"]), acc)
            is_correct = pred == ex["answer"]
            correct += int(is_correct)
            per_example.append({
                "sentence": ex["sentence"],
                "predicted": pred,
                "ground_truth": ex["answer"],
                "correct": is_correct,
            })

        overall = correct / config.num_examples if config.num_examples > 0 else 0.0
        return BenchmarkResult(
            benchmark_type=BenchmarkType.WINOGRANDE,
            model_name=self.model_name,
            overall_score=round(overall, 4),
            category_scores={"coreference_resolution": round(overall, 4)},
            num_examples=config.num_examples,
            num_correct=correct,
            elapsed_seconds=time.time() - start,
            config=config.to_dict(),
            per_example=per_example,
        )

    def run_all(
        self,
        num_examples: int = 100,
        seed: int = 42,
    ) -> Dict[BenchmarkType, BenchmarkResult]:
        """Run all supported benchmarks with common settings."""
        results: Dict[BenchmarkType, BenchmarkResult] = {}
        for bt in BenchmarkType:
            config = BenchmarkConfig(
                benchmark_type=bt,
                num_examples=num_examples,
                seed=seed,
            )
            runner_map = {
                BenchmarkType.MMLU: self.run_mmlu,
                BenchmarkType.HELLASWAG: self.run_hellaswag,
                BenchmarkType.ARC: self.run_arc,
                BenchmarkType.WINOGRANDE: self.run_winogrande,
            }
            results[bt] = runner_map[bt](config)
            logger.info(
                "%s: %.2f%% (%d/%d) in %.2fs",
                bt.value,
                results[bt].overall_score * 100,
                results[bt].num_correct,
                results[bt].num_examples,
                results[bt].elapsed_seconds,
            )
        return results
