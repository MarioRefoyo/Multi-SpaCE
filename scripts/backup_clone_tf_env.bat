@echo off
setlocal enabledelayedexpansion

set "BASE_ENV=counterfactuals_tf_gpu"
set "NEW_ENV=counterfactuals_tf_gpu_mascots"
set "BACKUP_DIR=env_backups"

if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"

echo Base environment: %BASE_ENV%
echo New environment:  %NEW_ENV%
echo Backup folder:    %BACKUP_DIR%
echo.

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

echo [1/5] Exporting full Conda environment...
call conda env export --name "%BASE_ENV%" > "%BACKUP_DIR%\%BASE_ENV%_%STAMP%.yml"
if errorlevel 1 exit /b 1

echo [2/5] Exporting explicit package lock...
call conda list --explicit --name "%BASE_ENV%" > "%BACKUP_DIR%\%BASE_ENV%_%STAMP%_explicit.txt"
if errorlevel 1 exit /b 1

echo [3/5] Exporting pip freeze...
call conda run --name "%BASE_ENV%" python -m pip freeze > "%BACKUP_DIR%\%BASE_ENV%_%STAMP%_pip_freeze.txt"
if errorlevel 1 exit /b 1

echo [4/5] Cloning environment...
call conda create --name "%NEW_ENV%" --clone "%BASE_ENV%" -y
if errorlevel 1 exit /b 1

echo [5/5] Verifying TensorFlow and GPU visibility in cloned environment...
call conda run --name "%NEW_ENV%" python -c "import sys, tensorflow as tf; print('python', sys.version); print('tensorflow', tf.__version__); print('gpus', tf.config.list_physical_devices('GPU'))"
if errorlevel 1 exit /b 1

echo.
echo Done.
echo Backup files written to: %BACKUP_DIR%
echo Cloned environment: %NEW_ENV%
