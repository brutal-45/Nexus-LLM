@echo off
REM ============================================================================
REM Nexus-LLM Quick Start Script (Windows)
REM ============================================================================
REM Activates the virtual environment and starts an interactive chat session
REM with the default model (gpt2-medium).
REM
REM Usage:
REM   scripts\start.bat [model_id]
REM
REM Examples:
REM   scripts\start.bat              - use default model (gpt2-medium)
REM   scripts\start.bat phi-2        - start with phi-2
REM   scripts\start.bat tinyllama    - start with TinyLlama
REM ============================================================================

setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "DEFAULT_MODEL=gpt2-medium"

REM Use first argument as model, or default
if "%~1"=="" (
    set "MODEL=%DEFAULT_MODEL%"
) else (
    set "MODEL=%~1"
)

REM ---------------------------------------------------------------------------
REM Activate virtual environment
REM ---------------------------------------------------------------------------
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at %VENV_DIR%
    echo         Run the install script first:
    echo           scripts\install.bat
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM ---------------------------------------------------------------------------
REM Launch chat
REM ---------------------------------------------------------------------------
echo Starting Nexus-LLM with model: %MODEL%
echo Press Ctrl+C to exit.
echo.

nexus-llm chat --model %MODEL%

endlocal
