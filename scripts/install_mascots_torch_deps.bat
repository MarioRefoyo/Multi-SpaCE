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

echo Installing CUDA-enabled PyTorch stack into:
echo   %ENV_PREFIX%
echo.
echo This uses the PyTorch CUDA 12.4 wheel, matching the torch build already used by this repository.
echo TensorFlow keeps using the Conda CUDA 11.2/cuDNN 8.1 runtime in the same environment.
echo.

call conda run --prefix "%ENV_PREFIX%" python -m pip install ^
    --index-url https://download.pytorch.org/whl/cu124 ^
    "torch==2.4.1+cu124"
if errorlevel 1 exit /b 1

call conda run --prefix "%ENV_PREFIX%" python -m pip install ^
    "pytorch-lightning==2.5.0.post0"
if errorlevel 1 exit /b 1

echo.
echo Verifying TensorFlow GPU and Torch CUDA in one process...
call conda run --prefix "%ENV_PREFIX%" python -c "import tensorflow as tf; import torch; print('tf', tf.__version__, tf.config.list_physical_devices('GPU')); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if errorlevel 1 exit /b 1

echo.
echo Done.
