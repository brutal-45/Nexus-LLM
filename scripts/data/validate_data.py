#!/usr/bin/env python3
"""
Validate Data Format - Nexus-LLM
=================================
Validates training data files for correct format, required fields,
and data quality issues before fine-tuning.

Usage:
    python validate_data.py --input data/train.jsonl --format chatml
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Schema definitions for each format
SCHEMAS = {
    "chatml": {
        "required_fields": ["messages"],
        "message_fields": ["role", "content"],
        "valid_roles": ["system", "user", "assistant"],
    },
    "alpaca": {
        "required_fields": ["instruction", "output"],
        "optional_fields": ["input"],
    },
    "sharegpt": {
        "required_fields": ["conversations"],
        "conversation_fields": ["from", "value"],
        "valid_from": ["system", "human", "gpt"],
    },
    "preference": {
        "required_fields": ["prompt", "chosen", "rejected"],
    },
}


class ValidationError:
    """Represents a single validation error."""

    def __init__(self, line: int, field: str, message: str, severity: str = "error"):
        self.line = line
        self.field = field
        self.message = message
        self.severity = severity

    def __str__(self):
        return f"[{self.severity.upper()}] Line {self.line}, field '{self.field}': {self.message}"


class DataValidator:
    """Validates training data against a schema."""

    def __init__(self, format_name: str, strict: bool = False):
        self.format_name = format_name
        self.schema = SCHEMAS.get(format_name)
        self.strict = strict
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.stats: Dict = Counter()

    def validate_file(self, file_path: str) -> bool:
        """Validate a JSONL file. Returns True if valid."""
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        if not path.suffix == ".jsonl":
            logger.warning(f"Expected .jsonl extension, got {path.suffix}")

        line_count = 0
        valid_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()

                if not line:
                    self.stats["empty_lines"] += 1
                    continue

                # Parse JSON
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    self.errors.append(ValidationError(
                        line=line_num,
                        field="json",
                        message=f"Invalid JSON: {e}",
                    ))
                    continue

                # Validate against schema
                if self._validate_item(item, line_num):
                    valid_count += 1

        self.stats["total_lines"] = line_count
        self.stats["valid_lines"] = valid_count
        self.stats["invalid_lines"] = line_count - valid_count

        return len(self.errors) == 0

    def _validate_item(self, item: Dict, line_num: int) -> bool:
        """Validate a single data item."""
        is_valid = True

        if self.format_name == "chatml":
            is_valid = self._validate_chatml(item, line_num)
        elif self.format_name == "alpaca":
            is_valid = self._validate_alpaca(item, line_num)
        elif self.format_name == "sharegpt":
            is_valid = self._validate_sharegpt(item, line_num)
        elif self.format_name == "preference":
            is_valid = self._validate_preference(item, line_num)
        else:
            # Generic validation
            is_valid = self._validate_generic(item, line_num)

        return is_valid

    def _validate_chatml(self, item: Dict, line_num: int) -> bool:
        """Validate ChatML format."""
        is_valid = True

        # Check required fields
        if "messages" not in item:
            self.errors.append(ValidationError(line_num, "messages", "Missing required field"))
            return False

        messages = item["messages"]
        if not isinstance(messages, list):
            self.errors.append(ValidationError(line_num, "messages", "Must be a list"))
            return False

        if len(messages) == 0:
            self.errors.append(ValidationError(line_num, "messages", "Messages list is empty"))
            return False

        # Validate each message
        valid_roles = SCHEMAS["chatml"]["valid_roles"]
        for i, msg in enumerate(messages):
            if "role" not in msg:
                self.errors.append(ValidationError(
                    line_num, f"messages[{i}].role", "Missing role field"))
                is_valid = False
            elif msg["role"] not in valid_roles:
                self.errors.append(ValidationError(
                    line_num, f"messages[{i}].role",
                    f"Invalid role '{msg['role']}', expected one of {valid_roles}",
                    severity="warning" if not self.strict else "error",
                ))

            if "content" not in msg:
                self.errors.append(ValidationError(
                    line_num, f"messages[{i}].content", "Missing content field"))
                is_valid = False
            elif not msg["content"].strip():
                self.warnings.append(ValidationError(
                    line_num, f"messages[{i}].content", "Empty content",
                    severity="warning"))

        # Check for at least one user and one assistant message
        roles = [msg.get("role") for msg in messages]
        if "user" not in roles:
            self.warnings.append(ValidationError(
                line_num, "messages", "No user message found", severity="warning"))
        if "assistant" not in roles:
            self.warnings.append(ValidationError(
                line_num, "messages", "No assistant message found", severity="warning"))

        # Track stats
        self.stats["total_messages"] += len(messages)
        self.stats["total_chars"] += sum(len(m.get("content", "")) for m in messages)

        return is_valid

    def _validate_alpaca(self, item: Dict, line_num: int) -> bool:
        """Validate Alpaca format."""
        is_valid = True

        for field in ["instruction", "output"]:
            if field not in item:
                self.errors.append(ValidationError(line_num, field, f"Missing required field"))
                is_valid = False
            elif not isinstance(item[field], str):
                self.errors.append(ValidationError(line_num, field, "Must be a string"))
                is_valid = False
            elif not item[field].strip():
                self.errors.append(ValidationError(line_num, field, "Empty string"))
                is_valid = False

        if "input" in item and not isinstance(item["input"], str):
            self.errors.append(ValidationError(line_num, "input", "Must be a string"))
            is_valid = False

        return is_valid

    def _validate_sharegpt(self, item: Dict, line_num: int) -> bool:
        """Validate ShareGPT format."""
        is_valid = True

        if "conversations" not in item:
            self.errors.append(ValidationError(line_num, "conversations", "Missing required field"))
            return False

        convs = item["conversations"]
        if not isinstance(convs, list) or len(convs) == 0:
            self.errors.append(ValidationError(line_num, "conversations", "Must be a non-empty list"))
            return False

        valid_from = SCHEMAS["sharegpt"]["valid_from"]
        for i, conv in enumerate(convs):
            if "from" not in conv:
                self.errors.append(ValidationError(line_num, f"conversations[{i}].from", "Missing from field"))
                is_valid = False
            elif conv["from"] not in valid_from:
                self.warnings.append(ValidationError(
                    line_num, f"conversations[{i}].from",
                    f"Unexpected 'from' value: {conv['from']}", severity="warning"))

            if "value" not in conv:
                self.errors.append(ValidationError(line_num, f"conversations[{i}].value", "Missing value field"))
                is_valid = False

        return is_valid

    def _validate_preference(self, item: Dict, line_num: int) -> bool:
        """Validate preference data format for DPO training."""
        is_valid = True

        for field in ["prompt", "chosen", "rejected"]:
            if field not in item:
                self.errors.append(ValidationError(line_num, field, f"Missing required field"))
                is_valid = False
            elif not isinstance(item[field], str):
                self.errors.append(ValidationError(line_num, field, "Must be a string"))
                is_valid = False
            elif not item[field].strip():
                self.errors.append(ValidationError(line_num, field, "Empty string"))
                is_valid = False

        # Check that chosen and rejected are different
        if "chosen" in item and "rejected" in item:
            if item["chosen"] == item["rejected"]:
                self.warnings.append(ValidationError(
                    line_num, "chosen/rejected",
                    "Chosen and rejected responses are identical", severity="warning"))

        return is_valid

    def _validate_generic(self, item: Dict, line_num: int) -> bool:
        """Generic validation for unknown formats."""
        if not isinstance(item, dict):
            self.errors.append(ValidationError(line_num, "", "Each line must be a JSON object"))
            return False
        if len(item) == 0:
            self.errors.append(ValidationError(line_num, "", "Empty object"))
            return False
        return True

    def print_report(self):
        """Print a validation report."""
        print("\n" + "=" * 60)
        print("Data Validation Report")
        print("=" * 60)
        print(f"Format: {self.format_name}")
        print(f"Total lines: {self.stats.get('total_lines', 0)}")
        print(f"Valid lines: {self.stats.get('valid_lines', 0)}")
        print(f"Invalid lines: {self.stats.get('invalid_lines', 0)}")
        print(f"Empty lines: {self.stats.get('empty_lines', 0)}")

        if self.stats.get("total_messages"):
            print(f"Total messages: {self.stats['total_messages']}")
            avg_chars = self.stats.get("total_chars", 0) / max(self.stats["total_messages"], 1)
            print(f"Average message length: {avg_chars:.0f} chars")

        if self.errors:
            print(f"\n{len(self.errors)} ERROR(S):")
            for error in self.errors[:20]:  # Show first 20
                print(f"  {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more")

        if self.warnings:
            print(f"\n{len(self.warnings)} WARNING(S):")
            for warning in self.warnings[:10]:
                print(f"  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        if not self.errors and not self.warnings:
            print("\nNo issues found. Data is valid!")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate training data for Nexus-LLM")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSONL file to validate")
    parser.add_argument("--format", type=str, default="chatml",
                       choices=["chatml", "alpaca", "sharegpt", "preference"],
                       help="Data format to validate against")
    parser.add_argument("--strict", action="store_true",
                       help="Treat warnings as errors")
    args = parser.parse_args()

    validator = DataValidator(format_name=args.format, strict=args.strict)
    is_valid = validator.validate_file(args.input)
    validator.print_report()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
