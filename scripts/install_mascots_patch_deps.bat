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

echo Installing/repairing MASCOTS patch dependencies in:
echo   %ENV_PREFIX%
echo.

call conda run --prefix "%ENV_PREFIX%" python -m pip install ^
    psutil ^
    numba==0.60.0 ^
    llvmlite==0.43.0 ^
    awkward ^
    sparse ^
    shap==0.42.1 ^
    loguru==0.7.2
if errorlevel 1 exit /b 1

echo.
echo Verifying imports from repository...
call conda run --prefix "%ENV_PREFIX%" python -c "from methods.MASCOTSCF import MASCOTSCF; import fast_borf; import mascots; print('MASCOTS wrapper import OK')"
if errorlevel 1 exit /b 1

echo.
echo Done.
