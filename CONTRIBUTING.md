# Contributing to Nexus-LLM

First off, thank you for considering contributing to Nexus-LLM! It's people like you that make Nexus-LLM such a great tool. 

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to nexus-llm@example.com.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nexus-llm.git
   cd nexus-llm
   ```
3. **Add** the upstream remote:
   ```bash
   git remote add upstream https://github.com/nexus-llm/nexus-llm.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Make (optional, for using Makefile targets)

### Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate on Windows

# Install in development mode with all dev dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
make pre-commit

# Or manually:
pre-commit install
```

## How to Contribute

### Bug Fixes

1. Check if the bug has been reported in [Issues](https://github.com/nexus-llm/nexus-llm/issues)
2. If not, create a new issue with the bug report template
3. Fork the repo and create a branch: `git checkout -b fix/bug-description`
4. Write a test that reproduces the bug
5. Fix the bug
6. Ensure all tests pass: `make test`
7. Submit a pull request

### Features

1. Open a feature request issue to discuss the proposed changes
2. Get approval from maintainers before starting work
3. Fork the repo and create a branch: `git checkout -b feature/feature-name`
4. Implement the feature with tests
5. Update documentation
6. Submit a pull request

## Pull Request Process

1. **Update documentation** - Ensure any new features are documented
2. **Add tests** - All new code should have corresponding tests
3. **Follow coding standards** - Run `make lint` and `make format`
4. **Keep PRs focused** - One feature/fix per PR
5. **Write clear descriptions** - Explain what the PR does and why
6. **Link related issues** - Reference any related issues in the PR description

### PR Checklist

- [ ] Code follows project style guidelines (`make lint` passes)
- [ ] Tests added for new functionality
- [ ] All tests pass (`make test`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No unnecessary files committed

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use **Black** for code formatting (line length: 120)
- Use **isort** for import sorting
- Use **type hints** for all function signatures
- Write **docstrings** for all public modules, classes, and functions

### Code Quality

```bash
# Run linters
make lint

# Auto-format code
make format
```

### Type Hints

All functions should have type annotations:

```python
def generate_text(prompt: str, max_tokens: int = 100) -> str:
    """Generate text from a prompt."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate the cross-entropy loss.

    Args:
        predictions: Model predictions of shape (batch_size, seq_len, vocab_size).
        targets: Target token ids of shape (batch_size, seq_len).

    Returns:
        The computed cross-entropy loss tensor.

    Raises:
        ValueError: If predictions and targets have incompatible shapes.
    """
    ...
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Build process or auxiliary tool changes
- `perf:` Performance improvements

Examples:
```
feat: add streaming support for chat command
fix: resolve memory leak in model loading
docs: update API documentation for server endpoints
test: add tests for data preprocessing pipeline
```

## Reporting Bugs

When filing a bug report, please include:

1. **Description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (Python version, OS, GPU/CPU, etc.)
6. **Logs** or error messages
7. **Possible fix** (if you have ideas)

## Feature Requests

When submitting a feature request, please include:

1. **Problem description** - What problem does this feature solve?
2. **Proposed solution** - How should this feature work?
3. **Alternatives considered** - What other approaches have you considered?
4. **Additional context** - Any other relevant information

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_cli.py -v

# Run tests matching a pattern
pytest tests/ -k "test_chat" -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

### Writing Tests

- Use **pytest** as the testing framework
- Place tests in the `tests/` directory mirroring the package structure
- Name test files `test_*.py`
- Write descriptive test function names
- Use fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`
- Mark GPU tests with `@pytest.mark.gpu`

Example:
```python
import pytest
from nexus_llm.types import Message, ChatRole


def test_message_creation():
    """Test that a Message object is created correctly."""
    msg = Message(role=ChatRole.USER, content="Hello")
    assert msg.role == ChatRole.USER
    assert msg.content == "Hello"


@pytest.mark.slow
def test_model_loading():
    """Test that a model can be loaded (slow)."""
    ...
```

## Documentation

- Update docstrings when changing code
- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes
- Use Google-style docstrings
- Build docs locally: `cd docs && make html`

## License

By contributing to Nexus-LLM, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
