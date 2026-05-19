.PHONY: install install-dev install-gpu test lint format clean run train serve chat download models eval benchmark config help

# Default target
help:
	@echo "Nexus-LLM Makefile Targets:"
	@echo ""
	@echo "  install        Install the package in development mode"
	@echo "  install-dev    Install with development dependencies"
	@echo "  install-gpu    Install with GPU support"
	@echo "  test           Run the test suite"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run all linters (flake8, mypy, isort check)"
	@echo "  format         Auto-format code (black, isort)"
	@echo "  clean          Remove build artifacts and caches"
	@echo "  run            Run the CLI"
	@echo "  train          Start training"
	@echo "  serve          Start the inference server"
	@echo "  chat           Start interactive chat"
	@echo "  download       Download a model"
	@echo "  models         List available models"
	@echo "  eval           Run evaluation"
	@echo "  benchmark      Run benchmarks"
	@echo "  config         View/edit configuration"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-up      Start Docker containers"
	@echo "  docker-down    Stop Docker containers"
	@echo "  pre-commit     Install and run pre-commit hooks"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

install-all:
	pip install -e ".[all]"

# Testing targets
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=nexus_llm --cov-report=html --cov-report=term-missing

test-quick:
	python -m pytest tests/ -v -x -k "not slow"

test-integration:
	python -m pytest tests/ -v -m integration

test-gpu:
	python -m pytest tests/ -v -m gpu

# Linting and formatting targets
lint:
	flake8 nexus_llm/ main.py
	mypy nexus_llm/
	isort --check-only --diff nexus_llm/ main.py
	black --check --diff nexus_llm/ main.py

format:
	black nexus_llm/ main.py
	isort nexus_llm/ main.py

format-check:
	black --check nexus_llm/ main.py
	isort --check-only nexus_llm/ main.py

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf nexus_llm.egg-info
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -f *.pyc
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-models:
	rm -rf models/

clean-logs:
	rm -rf logs/

# Run targets
run:
	python main.py --help

chat:
	python main.py chat

serve:
	python main.py serve

train:
	python main.py train --dataset ./data/train.jsonl

download:
	python main.py download gpt2-medium

models:
	python main.py models --list-available

eval:
	python main.py eval

benchmark:
	python main.py benchmark

config:
	python main.py config --list

# Docker targets
docker-build:
	docker build -t nexus-llm:latest .

docker-build-gpu:
	docker build --build-arg CUDA_VERSION=12.1 -t nexus-llm:gpu .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Development utilities
check-all: lint test
	@echo "All checks passed!"

setup-dev: install-dev pre-commit
	@echo "Development environment ready!"
