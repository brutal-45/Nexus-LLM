"""Tests for data validation (schema, completeness, quality scoring)."""
import pytest

from nexus_llm.utils.validation import (
    ValidationError,
    validate_type,
    validate_range,
    validate_path,
    validate_url,
    validate_choice,
    validate_non_empty,
    validate_email,
)


class TestValidateType:
    """Test type validation."""

    def test_valid_type_string(self):
        assert validate_type("hello", str) is True

    def test_valid_type_int(self):
        assert validate_type(42, int) is True

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="must be of type"):
            validate_type("hello", int)

    def test_invalid_type_no_raise(self):
        assert validate_type("hello", int, raise_error=False) is False

    def test_tuple_of_types(self):
        assert validate_type(3.14, (int, float)) is True


class TestValidateRange:
    """Test numeric range validation."""

    def test_value_within_range(self):
        assert validate_range(5, min_val=1, max_val=10) is True

    def test_value_below_min(self):
        with pytest.raises(ValidationError, match="must be >="):
            validate_range(0, min_val=1)

    def test_value_above_max(self):
        with pytest.raises(ValidationError, match="must be <="):
            validate_range(11, max_val=10)

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_range("abc", min_val=0)

    def test_exact_boundary(self):
        assert validate_range(10, min_val=10, max_val=10) is True


class TestValidatePath:
    """Test path validation."""

    def test_valid_path(self):
        assert validate_path("/tmp/test") is True

    def test_empty_path_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            validate_path("")

    def test_path_traversal_raises(self):
        with pytest.raises(ValidationError, match="path traversal"):
            validate_path("../../etc/passwd")

    def test_must_exist_nonexistent(self, tmp_dir):
        with pytest.raises(ValidationError, match="does not exist"):
            validate_path(str(tmp_dir / "nonexistent"), must_exist=True)

    def test_must_be_dir(self, tmp_dir):
        assert validate_path(str(tmp_dir), must_be_dir=True) is True


class TestValidateUrl:
    """Test URL validation."""

    def test_valid_http_url(self):
        assert validate_url("https://example.com") is True

    def test_url_without_scheme(self):
        with pytest.raises(ValidationError, match="must have a scheme"):
            validate_url("example.com")

    def test_restricted_scheme(self):
        with pytest.raises(ValidationError, match="must use one of"):
            validate_url("ftp://example.com", allowed_schemes=["http", "https"])

    def test_allowed_scheme_passes(self):
        assert validate_url("https://example.com", allowed_schemes=["http", "https"]) is True

    def test_empty_url_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            validate_url("")


class TestValidateChoice:
    """Test choice validation."""

    def test_valid_choice(self):
        assert validate_choice("auto", ["auto", "cpu", "cuda"]) is True

    def test_invalid_choice(self):
        with pytest.raises(ValidationError, match="must be one of"):
            validate_choice("tpu", ["auto", "cpu", "cuda"])


class TestValidateNonEmpty:
    """Test non-empty validation."""

    def test_non_empty_string(self):
        assert validate_non_empty("hello") is True

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            validate_non_empty("")

    def test_none_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            validate_non_empty(None)

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            validate_non_empty([])


class TestValidateEmail:
    """Test email validation."""

    def test_valid_email(self):
        assert validate_email("user@example.com") is True

    def test_invalid_email(self):
        with pytest.raises(ValidationError, match="not a valid email"):
            validate_email("not-an-email")

    def test_email_no_domain(self):
        with pytest.raises(ValidationError, match="not a valid email"):
            validate_email("user@")
