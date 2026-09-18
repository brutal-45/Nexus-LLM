# Contributing to Nexus-LLM

Thank you for your interest in contributing to Nexus-LLM! This guide outlines the process and standards for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive experience for everyone.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A CUDA-capable GPU (recommended for testing)
- Docker (optional, for containerized testing)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/Nexus-LLM.git
   cd Nexus-LLM
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   pip install pre-commit
   pre-commit install
   ```

4. **Verify the setup:**
   ```bash
   python -m pytest tests/ -x
   ```

## How to Contribute

### Reporting Bugs

1. Search existing issues to avoid duplicates.
2. Open a new issue using the **Bug Report** template.
3. Include: Python version, OS, GPU info, minimal reproduction steps, and error logs.

### Requesting Features

1. Open an issue using the **Feature Request** template.
2. Describe the use case, expected behavior, and any alternatives considered.

### Submitting Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or: git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards (see below).

3. **Write tests** for new functionality. Aim for >80% coverage on new code.

4. **Run the test suite:**
   ```bash
   # Unit tests
   python -m pytest tests/unit/ -v

   # Integration tests (requires GPU)
   python -m pytest tests/integration/ -v

   # Linting and formatting
   ruff check .
   ruff format --check .
   mypy nexus_llm/
   ```

5. **Commit with conventional commits:**
   ```
   feat: add streaming support for RAG pipelines
   fix: resolve memory leak in batch inference
   docs: update API endpoint documentation
   refactor: simplify plugin hook registration
   test: add tests for safety filter edge cases
   chore: update dependencies
   ```

6. **Push and open a Pull Request:**
   - Fill out the PR template completely.
   - Link related issues.
   - Ensure CI passes.
   - Request review from a maintainer.

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with line length of 100 characters.
- Use type hints for all public APIs.
- Use `ruff` for formatting and linting.
- Document all public classes and functions with docstrings (Google style).

### Example:

```python
def generate_response(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Response:
    """Generate a response from the model.

    Args:
        prompt: The input prompt string.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A Response object containing the generated text and metadata.

    Raises:
        ValueError: If temperature is outside the valid range.
    """
    ...
```

### Testing

- Write unit tests for all new logic in `tests/unit/`.
- Write integration tests for end-to-end features in `tests/integration/`.
- Use `pytest` fixtures for common setup.
- Mock external dependencies; do not require network access for unit tests.

### Documentation

- Update relevant `.md` files in `docs/` for user-facing changes.
- Add docstrings for all public APIs.
- Update the changelog for notable changes.

## Review Process

1. A maintainer will be assigned to review your PR.
2. Address review feedback by pushing additional commits.
3. Once approved, a maintainer will merge your PR.

## Release Process

Maintainers follow semantic versioning (SemVer):
- **Patch** (x.x.Z): Bug fixes, no API changes.
- **Minor** (x.Y.0): New features, backward compatible.
- **Major** (X.0.0): Breaking API changes.

## Questions?

Feel free to open a discussion on GitHub or reach out on our Discord server.
