"""Test model verification for Nexus-LLM."""
import hashlib
import pytest
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path


class VerificationError(Exception):
    pass


@dataclass
class VerificationResult:
    is_valid: bool
    model_name: str
    checks_passed: List[str]
    checks_failed: List[str]
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ModelVerifier:
    def __init__(self):
        self._expected_checksums: Dict[str, str] = {}

    def register_checksum(self, model_name: str, checksum: str):
        self._expected_checksums[model_name] = checksum

    def verify_checksum(self, data: bytes, expected: str) -> bool:
        actual = hashlib.sha256(data).hexdigest()
        return actual == expected

    def verify_file_integrity(self, file_path: str) -> bool:
        if not Path(file_path).exists():
            raise VerificationError(f"File not found: {file_path}")
        return True

    def verify_model_structure(self, files: List[str]) -> VerificationResult:
        required_files = ["config.json", "model.bin"]
        passed = []
        failed = []
        for req in required_files:
            if req in files:
                passed.append(f"has_{req}")
            else:
                failed.append(f"missing_{req}")
        return VerificationResult(
            is_valid=len(failed) == 0,
            model_name="checked",
            checks_passed=passed,
            checks_failed=failed,
        )

    def verify_config(self, config: dict) -> VerificationResult:
        passed = []
        failed = []
        if "model_type" in config:
            passed.append("has_model_type")
        else:
            failed.append("missing_model_type")
        if "vocab_size" in config:
            passed.append("has_vocab_size")
        else:
            failed.append("missing_vocab_size")
        if "hidden_size" in config:
            passed.append("has_hidden_size")
        else:
            failed.append("missing_hidden_size")
        return VerificationResult(
            is_valid=len(failed) == 0,
            model_name=config.get("name", "unknown"),
            checks_passed=passed,
            checks_failed=failed,
        )

    def verify_compatibility(self, model_config: dict, runtime_config: dict) -> VerificationResult:
        passed = []
        failed = []
        model_dtype = model_config.get("dtype", "float32")
        supported_dtypes = runtime_config.get("supported_dtypes", ["float32", "float16"])
        if model_dtype in supported_dtypes:
            passed.append("dtype_compatible")
        else:
            failed.append(f"dtype_incompatible:{model_dtype}")

        model_size = model_config.get("parameter_count", 0)
        max_size = runtime_config.get("max_parameters", float("inf"))
        if model_size <= max_size:
            passed.append("size_compatible")
        else:
            failed.append("size_exceeds_limit")

        return VerificationResult(
            is_valid=len(failed) == 0,
            model_name=model_config.get("name", "unknown"),
            checks_passed=passed,
            checks_failed=failed,
        )


class TestVerificationResult:
    def test_valid_result(self):
        result = VerificationResult(is_valid=True, model_name="test", checks_passed=["a"], checks_failed=[])
        assert result.is_valid is True
        assert len(result.checks_passed) == 1

    def test_invalid_result(self):
        result = VerificationResult(is_valid=False, model_name="test", checks_passed=[], checks_failed=["b"])
        assert result.is_valid is False

    def test_default_details(self):
        result = VerificationResult(is_valid=True, model_name="test", checks_passed=[], checks_failed=[])
        assert result.details == {}


class TestModelVerifier:
    def test_register_and_verify_checksum(self):
        verifier = ModelVerifier()
        data = b"test data"
        checksum = hashlib.sha256(data).hexdigest()
        verifier.register_checksum("test-model", checksum)
        assert verifier.verify_checksum(data, checksum) is True

    def test_checksum_mismatch(self):
        verifier = ModelVerifier()
        assert verifier.verify_checksum(b"test", "wrong_checksum") is False

    def test_verify_file_integrity_existing(self, tmp_dir):
        f = tmp_dir / "model.bin"
        f.write_bytes(b"data")
        verifier = ModelVerifier()
        assert verifier.verify_file_integrity(str(f)) is True

    def test_verify_file_integrity_missing(self):
        verifier = ModelVerifier()
        with pytest.raises(VerificationError, match="not found"):
            verifier.verify_file_integrity("/nonexistent/file")

    def test_verify_model_structure_valid(self):
        verifier = ModelVerifier()
        result = verifier.verify_model_structure(["config.json", "model.bin", "tokenizer.json"])
        assert result.is_valid is True

    def test_verify_model_structure_missing_files(self):
        verifier = ModelVerifier()
        result = verifier.verify_model_structure(["tokenizer.json"])
        assert result.is_valid is False
        assert len(result.checks_failed) >= 1

    def test_verify_config_valid(self):
        verifier = ModelVerifier()
        config = {"model_type": "gpt2", "vocab_size": 50257, "hidden_size": 768}
        result = verifier.verify_config(config)
        assert result.is_valid is True

    def test_verify_config_missing_fields(self):
        verifier = ModelVerifier()
        config = {"model_type": "gpt2"}
        result = verifier.verify_config(config)
        assert result.is_valid is False

    def test_verify_compatibility_compatible(self):
        verifier = ModelVerifier()
        model_cfg = {"dtype": "float32", "parameter_count": 7000000000}
        runtime_cfg = {"supported_dtypes": ["float32", "float16"], "max_parameters": 15000000000}
        result = verifier.verify_compatibility(model_cfg, runtime_cfg)
        assert result.is_valid is True

    def test_verify_compatibility_dtype_incompatible(self):
        verifier = ModelVerifier()
        model_cfg = {"dtype": "bfloat16", "parameter_count": 1000}
        runtime_cfg = {"supported_dtypes": ["float32"], "max_parameters": 100000}
        result = verifier.verify_compatibility(model_cfg, runtime_cfg)
        assert result.is_valid is False
