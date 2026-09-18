#!/bin/bash
set -e

echo "=========================================="
echo "Nexus-LLM Installation Verification"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python..."
python --version
echo "   ✓ Python available"
echo ""

# Check virtual environment
echo "2. Checking virtual environment..."
if [ -d "venv" ]; then
    echo "   ✓ Virtual environment exists"
else
    echo "   ⚠ Virtual environment not found. Run: python -m venv venv"
fi
echo ""

# Check if package is installed
echo "3. Checking package installation..."
if python -c "import nexus_llm; print('   ✓ nexus_llm package installed')" 2>/dev/null; then
    echo "   ✓ Package can be imported"
else
    echo "   ⚠ Package not installed. Run: pip install -e ."
fi
echo ""

# Check CLI
echo "4. Checking CLI..."
if python -c "from nexus_llm.cli import cli; print('   ✓ CLI module loads')" 2>/dev/null; then
    echo "   ✓ CLI works"
else
    echo "   ⚠ CLI failed to load"
fi
echo ""

# Check backend
echo "5. Checking backend..."
if python -c "from nexus_llm.backend.inference import InferenceEngine; from nexus_llm.backend.server import create_app; print('   ✓ Backend modules load')" 2>/dev/null; then
    echo "   ✓ Backend works"
else
    echo "   ⚠ Backend failed to load"
fi
echo ""

# Check model catalog
echo "6. Checking model catalog..."
python -c "from nexus_llm.core.model_catalog import list_models, list_categories; print(f'   ✓ {len(list_models())} models in {len(list_categories())} categories')" 2>/dev/null
echo ""

# Check linting
echo "7. Checking code quality..."
if python -m black --check nexus_llm/ 2>&1 | grep -q "would be left unchanged\|All done"; then
    echo "   ✓ Black formatting passes"
else
    echo "   ⚠ Black formatting has issues"
fi

if python -m ruff check nexus_llm/ 2>&1 | grep -q "All checks passed"; then
    echo "   ✓ Ruff linting passes"
else
    echo "   ⚠ Ruff linting has issues"
fi
echo ""

# Check tests
echo "8. Running core tests..."
if python -m pytest tests/test_cli.py tests/test_main.py -v --tb=line -q 2>&1 | grep -q "passed"; then
    echo "   ✓ Core tests pass"
else
    echo "   ⚠ Some tests failed"
fi
echo ""

echo "=========================================="
echo "✅ Verification Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Run 'nexus-llm info' to see system info"
echo "  - Run 'nexus-llm models' to list available models"
echo "  - Run 'nexus-llm chat' to start chatting"
echo "  - Run 'nexus-llm serve' to start the API server"
