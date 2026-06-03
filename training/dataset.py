"""Dataset Preparation - Load, process, and format training data."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """
    Prepares and processes datasets for training/fine-tuning.
    Supports loading from JSONL files, HuggingFace datasets,
    and conversation format conversion.
    """

    # Supported data formats
    SUPPORTED_FORMATS = ["jsonl", "json", "csv", "hf_dataset"]

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_format: str = "jsonl",
        split_ratio: float = 0.9,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.split_ratio = split_ratio
        self.max_samples = max_samples
        self.seed = seed
        self._dataset: Optional[Dataset] = None

    def load_from_jsonl(self, path: str) -> Dataset:
        """
        Load a dataset from a JSONL file.

        Expected format per line:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        or
        {"prompt": "...", "response": "..."}
        or
        {"text": "..."}
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    processed = self._normalize_item(item)
                    if processed:
                        data.append(processed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        if self.max_samples and len(data) > self.max_samples:
            data = data[:self.max_samples]

        self._dataset = Dataset.from_list(data)
        logger.info(f"Loaded {len(data)} examples from {path}")
        return self._dataset

    def load_from_huggingface(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
    ) -> Dataset:
        """Load a dataset from HuggingFace Hub."""
        kwargs = {"path": dataset_name, "split": split}
        if subset:
            kwargs["name"] = subset

        self._dataset = load_dataset(**kwargs)

        if self.max_samples and len(self._dataset) > self.max_samples:
            self._dataset = self._dataset.select(range(self.max_samples))

        logger.info(
            f"Loaded {len(self._dataset)} examples from HuggingFace: {dataset_name}"
        )
        return self._dataset

    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a data item to a standard format."""
        # Format 1: Chat messages format
        if "messages" in item:
            return {
                "messages": item["messages"],
                "text": self._messages_to_text(item["messages"]),
            }

        # Format 2: Prompt-response format
        if "prompt" in item and "response" in item:
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
            return {
                "messages": messages,
                "text": self._messages_to_text(messages),
            }

        # Format 3: Instruction format (Alpaca-style)
        if "instruction" in item:
            prompt = item["instruction"]
            if item.get("input"):
                prompt += f"\n\nInput: {item['input']}"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": item.get("output", "")},
            ]
            return {
                "messages": messages,
                "text": self._messages_to_text(messages),
            }

        # Format 4: Plain text
        if "text" in item:
            return {"text": item["text"], "messages": []}

        logger.warning(f"Unknown data format, skipping: {list(item.keys())}")
        return None

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages format to plain text for causal LM training."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def prepare_for_training(
        self,
        tokenizer=None,
        max_length: int = 1024,
        format_type: str = "causal",
    ) -> Dataset:
        """
        Prepare the dataset for training with tokenization.

        Args:
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            format_type: "causal" for causal LM, "chat" for chat format

        Returns:
            Tokenized dataset ready for training
        """
        if self._dataset is None:
            raise ValueError("No dataset loaded. Call load_from_* first.")

        def tokenize_function(examples):
            if tokenizer is None:
                return examples

            texts = examples.get("text", [])
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            # For causal LM, labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = self._dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self._dataset.column_names,
            desc="Tokenizing dataset",
        )

        # Split into train/validation
        if self.split_ratio < 1.0:
            split = tokenized_dataset.train_test_split(
                test_size=1 - self.split_ratio,
                seed=self.seed,
            )
            logger.info(
                f"Train: {len(split['train'])} examples, "
                f"Validation: {len(split['test'])} examples"
            )
            return split

        return tokenized_dataset

    def create_sample_dataset(self, output_path: str = "./data/training_data.jsonl") -> str:
        """
        Create a sample training dataset for testing purposes.
        This provides example data in the expected format.
        """
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I read a file in Python?"},
                    {"role": "assistant", "content": "You can read a file in Python using the `open()` function:\n\n```python\nwith open('filename.txt', 'r') as f:\n    content = f.read()\n```\n\nThe `with` statement ensures the file is properly closed after reading."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain what a function is in programming."},
                    {"role": "assistant", "content": "A function is a reusable block of code that performs a specific task. Functions help organize code, reduce repetition, and make programs easier to understand and maintain. In Python, you define a function using the `def` keyword:\n\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n```"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is the difference between a list and a tuple?"},
                    {"role": "assistant", "content": "The main differences between lists and tuples in Python are:\n\n1. **Mutability**: Lists are mutable (can be changed), tuples are immutable (cannot be changed)\n2. **Syntax**: Lists use square brackets `[]`, tuples use parentheses `()`\n3. **Performance**: Tuples are slightly faster than lists\n4. **Use case**: Use lists when you need a dynamic collection, tuples for fixed collections"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How does machine learning work?"},
                    {"role": "assistant", "content": "Machine learning works by training algorithms on data to find patterns and make predictions. The process involves:\n\n1. **Data Collection**: Gathering relevant training data\n2. **Feature Engineering**: Selecting important data attributes\n3. **Model Training**: The algorithm learns patterns from the data\n4. **Evaluation**: Testing the model on unseen data\n5. **Prediction**: Using the trained model to make new predictions\n\nThe key idea is that the model improves its performance through experience (more data)."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Write a function to reverse a string in Python."},
                    {"role": "assistant", "content": "Here are a few ways to reverse a string in Python:\n\n```python\n# Method 1: Slicing (most Pythonic)\ndef reverse_string(s):\n    return s[::-1]\n\n# Method 2: Using reversed()\ndef reverse_string_v2(s):\n    return ''.join(reversed(s))\n\n# Method 3: Using a loop\ndef reverse_string_v3(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result\n```\n\nThe slicing method `s[::-1]` is the most concise and Pythonic way."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is an API?"},
                    {"role": "assistant", "content": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. It defines the methods and data formats that applications can use to request and exchange information. Think of it as a waiter in a restaurant - it takes your order (request) to the kitchen (system) and brings back your food (response)."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain recursion with an example."},
                    {"role": "assistant", "content": "Recursion is when a function calls itself to solve a problem by breaking it down into smaller, similar sub-problems. Every recursive function needs:\n\n1. **Base case**: The condition that stops recursion\n2. **Recursive case**: The part where the function calls itself\n\nExample - Factorial:\n```python\ndef factorial(n):\n    if n <= 1:  # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive case\n```\n\nFor `factorial(5)`: 5 * 4 * 3 * 2 * 1 = 120"},
                ]
            },
        ]

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Sample dataset created at: {output_path}")
        return output_path

    @property
    def dataset(self) -> Optional[Dataset]:
        """Get the current dataset."""
        return self._dataset

    @property
    def dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset."""
        if self._dataset is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "num_examples": len(self._dataset),
            "columns": self._dataset.column_names,
            "features": str(self._dataset.features),
        }
