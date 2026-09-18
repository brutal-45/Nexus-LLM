@echo off
REM ============================================================================
REM Nexus-LLM Installation Script (Windows)
REM ============================================================================
REM This script sets up the Nexus-LLM environment:
REM   - Checks Python version (3.9+)
REM   - Creates a virtual environment
REM   - Installs dependencies
REM   - Downloads the default model (gpt2-medium)
REM   - Creates necessary directories
REM
REM Usage:
REM   scripts\install.bat
REM ============================================================================

setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------
REM Project root detection
REM ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "DEFAULT_MODEL=gpt2-medium"

REM ---------------------------------------------------------------------------
REM Banner
REM ---------------------------------------------------------------------------
echo.
echo ========================================
echo   Nexus-LLM Installer v2.0.0
echo ========================================
echo.

REM ---------------------------------------------------------------------------
REM Step 1: Check Python version
REM ---------------------------------------------------------------------------
[INFO]  Checking Python version...

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.9+: https://www.python.org/downloads/
    exit /b 1
)

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ('python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if !PY_MAJOR! lss 3 (
    echo [ERROR] Python 3.9+ is required, but found Python !PY_MAJOR!.!PY_MINOR!.
    echo         Please upgrade: https://www.python.org/downloads/
    exit /b 1
)

if !PY_MAJOR! equ 3 if !PY_MINOR! lss 9 (
    echo [ERROR] Python 3.9+ is required, but found Python 3.!PY_MINOR!.
    echo         Please upgrade: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "PY_VERSION=%%v"
echo [OK]    Python !PY_VERSION! detected.

REM ---------------------------------------------------------------------------
REM Step 2: Create virtual environment
REM ---------------------------------------------------------------------------
echo [INFO]  Creating virtual environment at %VENV_DIR%...

if exist "%VENV_DIR%" (
    echo [WARN]  Virtual environment already exists. Removing and recreating...
    rmdir /s /q "%VENV_DIR%"
)

python -m venv "%VENV_DIR%"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

echo [OK]    Virtual environment created.

REM Activate the venv for the rest of the script
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip and install wheel
echo [INFO]  Upgrading pip and installing build tools...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [OK]    Build tools updated.

REM ---------------------------------------------------------------------------
REM Step 3: Install dependencies
REM ---------------------------------------------------------------------------
echo [INFO]  Installing Nexus-LLM dependencies...
echo         This may take a few minutes on first run...

if exist "%PROJECT_ROOT%\pyproject.toml" (
    python -m pip install -e "%PROJECT_ROOT%"
) else if exist "%PROJECT_ROOT%\requirements.txt" (
    python -m pip install -r "%PROJECT_ROOT%\requirements.txt"
) else (
    echo [ERROR] No pyproject.toml or requirements.txt found.
    call deactivate
    exit /b 1
)

echo [OK]    Dependencies installed.

REM ---------------------------------------------------------------------------
REM Step 4: Create necessary directories
REM ---------------------------------------------------------------------------
echo [INFO]  Creating project directories...

set "DIRS=models data logs config checkpoints"

for %%d in (%DIRS%) do (
    set "DIRPATH=%PROJECT_ROOT%\%%d"
    if not exist "!DIRPATH!" (
        mkdir "!DIRPATH!"
        echo   created  !DIRPATH!
    ) else (
        echo   exists   !DIRPATH!
    )
)

echo [OK]    Directories ready.

REM ---------------------------------------------------------------------------
REM Step 5: Download the default model
REM ---------------------------------------------------------------------------
echo [INFO]  Downloading default model (%DEFAULT_MODEL%)...
echo         This downloads ~1.5 GB from HuggingFace Hub.
echo         You can skip this with Ctrl+C and download later.

python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('openai-community/%DEFAULT_MODEL%')" 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK]    Default model '%DEFAULT_MODEL%' downloaded and cached.
) else (
    echo [WARN]  Model download failed or was skipped.
    echo         You can download it later with:
    echo           python scripts\download_model.py %DEFAULT_MODEL%
)

REM ---------------------------------------------------------------------------
REM Step 6: Deactivate venv
REM ---------------------------------------------------------------------------
call deactivate

REM ---------------------------------------------------------------------------
REM Success!
REM ---------------------------------------------------------------------------
echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo   Nexus-LLM is ready to use!
echo.
echo   Next steps:
echo     1. Activate the virtual environment:
echo        .venv\Scripts\activate
echo.
echo     2. Start chatting:
echo        nexus-llm chat
echo        nexus-llm chat --model gpt2-medium
echo.
echo     3. Or use the quick-start script:
echo        scripts\start.bat
echo.
echo     4. Download more models:
echo        nexus-llm download phi-2
echo        nexus-llm models
echo.
echo     5. Start the API server:
echo        nexus-llm serve --port 8000
echo.
echo   Documentation: https://github.com/brutal-45/Nexus-LLM
echo.

endlocal
