# ============================================================================
# Nexus-LLM Makefile
# ============================================================================
# Common development tasks. Adjust VENV_DIR if you use a different location.
#
# Usage:
#   make install     - Set up the project (venv + dependencies)
#   make run         - Start interactive chat with default model
#   make serve       - Start the API server
#   make train       - Fine-tune a model (set MODEL and DATA)
#   make test        - Run the test suite
#   make clean       - Remove build artifacts and caches
#   make download    - Download a model (set MODEL)
# ============================================================================

VENV_DIR   := .venv
PYTHON     := $(VENV_DIR)/bin/python
PIP        := $(VENV_DIR)/bin/pip
MODEL      ?= gpt2-medium
DATA       ?= data/train.jsonl
HOST       ?= 127.0.0.1
PORT       ?= 8000

# ---------------------------------------------------------------------------
# install — Create venv and install dependencies
# ---------------------------------------------------------------------------
.PHONY: install
install: $(VENV_DIR)/bin/activate
	@echo "✓ Nexus-LLM installed. Run 'make run' to start."

$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@mkdir -p models data logs config checkpoints

# ---------------------------------------------------------------------------
# run — Start interactive chat
# ---------------------------------------------------------------------------
.PHONY: run
run: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/nexus-llm chat --model $(MODEL)

# ---------------------------------------------------------------------------
# serve — Start the API server
# ---------------------------------------------------------------------------
.PHONY: serve
serve: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/nexus-llm serve --host $(HOST) --port $(PORT)

# ---------------------------------------------------------------------------
# train — Fine-tune a model
# ---------------------------------------------------------------------------
.PHONY: train
train: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/nexus-llm train -m $(MODEL) -d $(DATA)

# ---------------------------------------------------------------------------
# test — Run the test suite
# ---------------------------------------------------------------------------
.PHONY: test
test: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
# clean — Remove build artifacts, caches, and temporary files
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf .venv
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf nexus_llm/__pycache__ nexus_llm/**/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned build artifacts and caches."

# ---------------------------------------------------------------------------
# download — Download a model from HuggingFace
# ---------------------------------------------------------------------------
.PHONY: download
download: $(VENV_DIR)/bin/activate
	$(PYTHON) scripts/download_model.py $(MODEL)
