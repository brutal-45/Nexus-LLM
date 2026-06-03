# Nexus-LLM Installation Guide

This guide walks you through installing Nexus-LLM from scratch. Whether you're setting up on a laptop for experimentation or a GPU server for production inference, we've got you covered.

---

## 1. Prerequisites

Before installing Nexus-LLM, ensure your system meets the following requirements:

### Required Software

- **Python 3.9 or later** — Nexus-LLM relies on modern Python features including type hints, dataclasses, and the `match` statement (Python 3.10+). We recommend Python 3.10 or 3.11 for the best compatibility and performance. You can check your Python version by running `python --version`. If your system Python is older, consider using [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/) to manage multiple Python versions.

- **pip 21.0 or later** — The package installer must support the `manylinux` wheel format and dependency resolution. Update pip with `pip install --upgrade pip` if needed.

- **git 2.0 or later** — Required for cloning the repository and for the optional development setup. Install from your system's package manager (e.g., `apt install git`, `brew install git`, or the [official Git website](https://git-scm.com/)).

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 16 GB+ |
| Disk Space | 2 GB | 20 GB+ (for models) |
| CPU | 2 cores | 4+ cores |
| GPU | None (CPU mode) | NVIDIA GPU with 8GB+ VRAM |

### Operating System Support

- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+ (full support)
- **macOS**: 12.0+ Monterey (Apple Silicon and Intel)
- **Windows**: Windows 10/11 with WSL2 (native Windows is experimental)

---

## 2. Quick Install

Get Nexus-LLM up and running in three commands. This method installs the latest stable release from PyPI and is recommended for most users.

```bash
# Step 1: Create a virtual environment (recommended)
python -m venv nexus-env
source nexus-env/bin/activate  # On Windows: nexus-env\Scripts\activate

# Step 2: Install Nexus-LLM
pip install nexus-llm

# Step 3: Verify the installation
nexus-llm --version
```

That's it! The `pip install` command will automatically pull in all required dependencies including PyTorch, Transformers, FastAPI, and Rich. If you already have PyTorch installed with CUDA support, pip will respect that and not overwrite it.

### Installing from Source

If you want the latest development version or plan to contribute to the project, install from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/brutal-45/Nexus-LLM.git
cd nexus-llm

# Install in development mode
pip install -e ".[dev]"

# Verify
nexus-llm --version
```

The `-e` flag installs in editable mode, meaning any changes you make to the source code are immediately reflected without reinstalling. The `[dev]` extra installs development tools like pytest, black, mypy, and pre-commit hooks.

### Installing Specific Extras

Nexus-LLM offers optional dependency groups for different use cases:

```bash
pip install "nexus-llm[training]"    # Fine-tuning dependencies (PEFT, datasets)
pip install "nexus-llm[server]"      # Production server dependencies (uvicorn, gunicorn)
pip install "nexus-llm[all]"         # Everything
```

---

## 3. GPU Setup (CUDA)

Running Nexus-LLM on a GPU dramatically speeds up inference — often 10-50x faster than CPU. Here's how to set up CUDA support properly.

### Check GPU Compatibility

First, verify you have a compatible NVIDIA GPU:

```bash
nvidia-smi
```

This should display your GPU model, driver version, and CUDA version. If the command is not found, you need to install NVIDIA drivers first. Nexus-LLM requires CUDA 11.8 or later.

### Install PyTorch with CUDA

The most common installation issue is having the CPU-only version of PyTorch. To ensure CUDA support, install PyTorch with the correct CUDA version **before** installing Nexus-LLM:

```bash
# For CUDA 12.1 (recommended for most modern GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (for older GPUs or drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install Nexus-LLM:

```bash
pip install nexus-llm
```

### Verify GPU Detection

After installation, confirm that PyTorch can see your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
```

If you see `CUDA available: True` and your GPU name, you're all set. Nexus-LLM will automatically use the GPU when available.

### Apple Silicon (M1/M2/M3/M4)

Apple Silicon Macs use Metal Performance Shaders (MPS) for GPU acceleration:

```bash
pip install torch torchvision torchaudio
```

PyTorch on Apple Silicon automatically uses MPS when available. Verify with:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Multi-GPU Setup

For servers with multiple GPUs, Nexus-LLM automatically detects all available GPUs. You can specify which GPU to use:

```bash
# Use GPU 0
nexus-llm --device cuda:0

# Use GPU 1
nexus-llm --device cuda:1

# Let Nexus-LLM auto-select (uses GPU with most free memory)
nexus-llm --device auto
```

---

## 4. Verify Installation

After installing Nexus-LLM, run through these verification steps to make sure everything is working correctly.

### Check CLI

```bash
# Display version
nexus-llm --version

# List available commands
nexus-llm --help

# Show configuration
nexus-llm config --show
```

### Run the Diagnostic

Nexus-LLM includes a built-in diagnostic that checks your environment:

```bash
nexus-llm doctor
```

This command checks:
- Python version compatibility
- PyTorch installation and CUDA availability
- Available disk space for model downloads
- Required dependencies
- Network connectivity (for model downloads)
- Configuration file integrity

### Quick Smoke Test

Load a small model and generate a response to confirm end-to-end functionality:

```bash
nexus-llm generate --model gpt2 --prompt "Hello, world!" --max-length 50
```

If you see generated text, everything is working! The first run will download the GPT-2 model (~500MB), which is cached locally for future use.

### Run the Test Suite (Source Install Only)

If you installed from source, run the test suite:

```bash
pytest tests/ -v
```

All tests should pass. If any fail, check the error messages for specific dependency or configuration issues.

---

## 5. First Run

Now that Nexus-LLM is installed and verified, let's start using it. There are two primary ways to interact with Nexus-LLM: the terminal chat interface and the API server.

### Terminal Chat

The easiest way to get started is the interactive terminal chat:

```bash
nexus-llm chat
```

This launches the default model (GPT-2 Medium) in an interactive terminal session with a beautiful Rich-based UI. You'll see:
- Syntax-highlighted responses with markdown rendering
- Token count and generation timing for each response
- Command history (use up/down arrows)
- Built-in commands (type `/help` for a list)

### Specify a Model

```bash
# Use a specific model
nexus-llm chat --model phi-2

# Use a model from a local path
nexus-llm chat --model ./models/my-model

# Use a HuggingFace model ID directly
nexus-llm chat --model microsoft/phi-2
```

### Start the API Server

To use Nexus-LLM as a backend service:

```bash
nexus-llm serve
```

The server starts on `http://127.0.0.1:8000` by default. You can then send requests:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in one paragraph", "max_length": 200}'
```

### Customize Configuration

Nexus-LLM is configured through a YAML file. View and modify settings:

```bash
# Show current configuration
nexus-llm config --show

# Set a specific value
nexus-llm config --set model.temperature 0.8
nexus-llm config --set server.port 8080

# Reset to defaults
nexus-llm config --reset
```

---

## 6. Common Issues and Fixes

### Issue: "CUDA out of memory"

**Symptoms**: The model fails to load or crashes during generation with a CUDA OOM error.

**Fixes**:
- Use a smaller model (e.g., `gpt2` instead of `gpt2-xl`)
- Reduce `max_length` in your configuration
- Enable 8-bit quantization: `nexus-llm chat --model gpt2-xl --load-in-8bit`
- Enable 4-bit quantization: `nexus-llm chat --model gpt2-xl --load-in-4bit`
- Set `precision: fp16` in your config to halve memory usage
- Close other GPU-intensive applications

### Issue: "Model download fails / Connection timeout"

**Symptoms**: `ConnectionError`, `TimeoutError`, or partial downloads when loading a model.

**Fixes**:
- Check your internet connection
- Set a HuggingFace mirror if in a restricted region: `export HF_ENDPOINT=https://hf-mirror.com`
- Download models manually: `huggingface-cli download gpt2-medium`
- Use a local model path: `nexus-llm chat --model /path/to/local/model`
- Set a custom cache directory: `nexus-llm config --set model.cache_dir /path/to/cache`

### Issue: "ModuleNotFoundError" after install

**Symptoms**: Python cannot find `nexus_llm` or its dependencies.

**Fixes**:
- Make sure you activated your virtual environment: `source nexus-env/bin/activate`
- Reinstall: `pip install --force-reinstall nexus-llm`
- Check for conflicting packages: `pip check`
- If using conda, ensure pip is installed in the conda environment: `conda install pip`

### Issue: "Slow generation on GPU"

**Symptoms**: GPU generation is only slightly faster than CPU or seems unreasonably slow.

**Fixes**:
- Verify CUDA is actually being used: `nexus-llm doctor`
- Check GPU utilization during generation: `watch -n 1 nvidia-smi`
- Ensure you're not accidentally running in CPU mode: `nexus-llm chat --device cuda`
- Update NVIDIA drivers to the latest version
- Make sure PyTorch is the CUDA version (not CPU): `python -c "import torch; print(torch.version.cuda)"`

### Issue: "Permission denied" on config or log files

**Symptoms**: Cannot write to configuration or log directories.

**Fixes**:
- Check file permissions: `ls -la ~/.config/nexus-llm/`
- Set a writable config path: `export NEXUS_CONFIG_DIR=/path/to/writable/dir`
- Run with appropriate permissions (avoid using `sudo` with pip)

### Issue: "Port already in use" when starting server

**Symptoms**: `OSError: [Errno 98] Address already in use` when running `nexus-llm serve`.

**Fixes**:
- Use a different port: `nexus-llm serve --port 8080`
- Find and kill the process using the port: `lsof -i :8000` then `kill <PID>`
- Set the port in config: `nexus-llm config --set server.port 8080`

---

## 7. Uninstall

To completely remove Nexus-LLM from your system:

### Uninstall the Package

```bash
pip uninstall nexus-llm
```

This removes the Python package and all its installed dependencies. If you want to also remove dependencies that may be shared with other packages, you can use:

```bash
pip autoremove  # If you have pip-autoremove installed
```

### Remove Configuration and Cache

Nexus-LLM stores configuration, cached models, and logs in several locations:

```bash
# Remove configuration
rm -rf ~/.config/nexus-llm/

# Remove cached models (WARNING: this deletes downloaded model weights!)
rm -rf ~/.cache/huggingface/hub/models--*

# Remove logs
rm -rf ./logs/nexus_llm.log

# Remove chat history
rm -f .nexus_history
```

### Remove Virtual Environment

If you created a dedicated virtual environment:

```bash
deactivate
rm -rf nexus-env/
```

### Remove Source Installation

If you installed from source:

```bash
rm -rf nexus-llm/  # The cloned repository
```

### Verify Removal

Confirm everything is cleaned up:

```bash
which nexus-llm        # Should return nothing
pip show nexus-llm     # Should return "not found"
```

If you have any issues with uninstallation, consult the [GitHub Issues](https://github.com/brutal-45/Nexus-LLM/issues) page for help.
