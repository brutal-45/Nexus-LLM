# ============================================================================
# Nexus-LLM Makefile
# ============================================================================
# Targets here mirror what CI runs, so `make check` reproduces the workflows
# locally. Adjust VENV_DIR if you use a different location.
#
#   make install     - Create .venv and install the package (runtime deps)
#   make install-dev - Editable install with dev + train extras (what CI uses)
#   make run         - Start interactive chat with the default model
#   make serve       - Start the API server
#   make train       - Fine-tune a model (set MODEL and DATA)
#   make test        - Run the test suite
#   make lint        - ruff + black --check
#   make format      - ruff --fix + black + isort
#   make typecheck   - mypy
#   make check       - lint + typecheck + tests (the CI gate)
#   make build       - Build sdist + wheel into dist/
#   make docker      - Build the container image
#   make clean       - Remove build artefacts and caches
# ============================================================================

VENV_DIR   ?= .venv
PYTHON     := $(VENV_DIR)/bin/python
PIP        := $(VENV_DIR)/bin/pip
PYTEST     := $(VENV_DIR)/bin/pytest
RUFF       := $(VENV_DIR)/bin/ruff
BLACK      := $(VENV_DIR)/bin/black
MYPY       := $(VENV_DIR)/bin/mypy

MODEL      ?= gpt2-medium
DATA       ?= data/datasets/alpaca_format.jsonl
HOST       ?= 127.0.0.1
PORT       ?= 8000
TESTS      ?= tests/
EXTRAS     ?= dev,train

# Python interpreter used to *create* the venv.
SYS_PYTHON ?= python3

# ---------------------------------------------------------------------------
# install — Create the venv and install the project
# ---------------------------------------------------------------------------
.PHONY: install
install: $(VENV_DIR)/bin/python
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install .
	@mkdir -p models data logs checkpoints
	@echo "✓ Nexus-LLM installed into $(VENV_DIR). Run 'make run' to start."

.PHONY: install-dev
install-dev: $(VENV_DIR)/bin/python
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[$(EXTRAS)]"
	$(VENV_DIR)/bin/pre-commit install || true
	@mkdir -p models data logs checkpoints
	@echo "✓ Development environment ready. Run 'make check' before pushing."

# Editable installs of this project must not be built by an ancient pip, so
# bootstrap the venv only once.
$(VENV_DIR)/bin/python:
	@$(SYS_PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else ("Python >= 3.9 is required, found %s" % sys.version.split()[0]))'
	$(SYS_PYTHON) -m venv $(VENV_DIR)
	@touch $(VENV_DIR)/bin/python

# ---------------------------------------------------------------------------
# run / serve / train — everyday entry points
# ---------------------------------------------------------------------------
.PHONY: run
run: $(VENV_DIR)/bin/python
	$(VENV_DIR)/bin/nexus-llm chat --model $(MODEL)

.PHONY: serve
serve: $(VENV_DIR)/bin/python
	$(VENV_DIR)/bin/nexus-llm serve --host $(HOST) --port $(PORT)

.PHONY: train
train: $(VENV_DIR)/bin/python
	$(VENV_DIR)/bin/nexus-llm train -m $(MODEL) -d $(DATA)

.PHONY: download
download: $(VENV_DIR)/bin/python
	$(PYTHON) scripts/download_model.py $(MODEL)

# ---------------------------------------------------------------------------
# test / lint / check — the CI gates
# ---------------------------------------------------------------------------
.PHONY: test
test: $(VENV_DIR)/bin/python
	$(PYTEST) $(TESTS) -v --tb=short

.PHONY: test-cov
test-cov: $(VENV_DIR)/bin/python
	$(PYTEST) $(TESTS) --cov=nexus_llm --cov-report=term-missing --cov-report=xml:coverage.xml

.PHONY: lint
lint: $(VENV_DIR)/bin/python
	$(RUFF) check nexus_llm/ scripts/
	$(BLACK) --check nexus_llm/

.PHONY: lint-fix
lint-fix: $(VENV_DIR)/bin/python
	$(RUFF) check --fix nexus_llm/ scripts/
	$(BLACK) nexus_llm/ scripts/

.PHONY: format
format: $(VENV_DIR)/bin/python
	$(BLACK) nexus_llm/ scripts/
	$(VENV_DIR)/bin/isort nexus_llm/ scripts/
	$(RUFF) check --fix nexus_llm/ scripts/

.PHONY: typecheck
typecheck: $(VENV_DIR)/bin/python
	$(MYPY) nexus_llm/

.PHONY: check-imports
check-imports: $(VENV_DIR)/bin/python
	$(PYTHON) scripts/dev/check_imports.py --strict

.PHONY: check-version
check-version: $(VENV_DIR)/bin/python
	$(PYTHON) scripts/dev/sync_version.py --check

.PHONY: check
check: lint check-imports check-version test
	@echo "✓ All CI checks passed."

# ---------------------------------------------------------------------------
# build / docker — packaging
# ---------------------------------------------------------------------------
.PHONY: build
build: $(VENV_DIR)/bin/python
	$(PIP) install --upgrade build twine
	$(PYTHON) -m build
	@echo "✓ Distributions in dist/ (verify with: make verify-dist)"

.PHONY: verify-dist
verify-dist: $(VENV_DIR)/bin/python
	$(PYTHON) -m twine check dist/*

.PHONY: docker
docker:
	docker build -t nexus-llm:latest .

.PHONY: docker-gpu
docker-gpu:
	docker build --build-arg CUDA_VERSION=12.1 -t nexus-llm:gpu .

.PHONY: docs
docs: $(VENV_DIR)/bin/python
	$(PIP) install -e ".[docs]"
	sphinx-build -b html docs docs/_build/html
	@echo "✓ Docs in docs/_build/html/index.html"

# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf build dist *.egg-info .eggs $(VENV_DIR) .tox
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
	rm -rf docs/_build site
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned build artefacts and caches."

.PHONY: help
help:
	@echo "Nexus-LLM targets:"
	@echo "  install        Create $(VENV_DIR) and install runtime deps"
	@echo "  install-dev    Editable install with [$(EXTRAS)] extras (what CI uses)"
	@echo "  run            Interactive chat (MODEL=$(MODEL))"
	@echo "  serve          API server on $(HOST):$(PORT)"
	@echo "  train          Fine-tune MODEL on DATA"
	@echo "  download       Fetch MODEL via scripts/download_model.py"
	@echo "  test / test-cov Run pytest (with coverage)"
	@echo "  lint / format   ruff + black (--fix / write)"
	@echo "  typecheck       mypy"
	@echo "  check-imports   Import every module; catches broken imports CI would miss"
	@echo "  check-version   VERSION / CHANGELOG consistency"
	@echo "  check           lint + import + version + tests  (the CI gate)"
	@echo "  build           sdist + wheel into dist/"
	@echo "  docker / docker-gpu  Build the container image"
	@echo "  docs            Build the Sphinx documentation"
	@echo "  clean           Remove build artefacts and caches
