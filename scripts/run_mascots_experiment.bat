@echo off
setlocal

set "ENV_NAME=mascots310"
set "WORK_DRIVE=%~d0"
set "ENV_PREFIX=%WORK_DRIVE%\.conda_envs\%ENV_NAME%"

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

if not exist "%ENV_PREFIX%" (
    echo ERROR: Environment not found: %ENV_PREFIX%
    exit /b 1
)

call conda run --prefix "%ENV_PREFIX%" python -m experiments.mascots
