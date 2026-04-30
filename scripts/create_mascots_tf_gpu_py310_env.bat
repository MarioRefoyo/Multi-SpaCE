@echo off
setlocal

set "ENV_NAME=mascots310"
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "WORK_DRIVE=%~d0"
set "ENV_PREFIX=%WORK_DRIVE%\.conda_envs\%ENV_NAME%"
set "CONDA_PKGS_DIRS=%WORK_DRIVE%\.conda_pkgs"
set "PIP_CACHE_DIR=%WORK_DRIVE%\.pip_cache"
set "TMP=%WORK_DRIVE%\.tmp"
set "TEMP=%WORK_DRIVE%\.tmp"

if not exist "%WORK_DRIVE%\.conda_envs" mkdir "%WORK_DRIVE%\.conda_envs"
if not exist "%CONDA_PKGS_DIRS%" mkdir "%CONDA_PKGS_DIRS%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

echo Creating experimental environment: %ENV_NAME%
echo Environment path: %ENV_PREFIX%
echo Conda package cache: %CONDA_PKGS_DIRS%
echo Pip cache: %PIP_CACHE_DIR%
echo This keeps counterfactuals_tf_gpu and counterfactuals_tf_gpu_mascots untouched.
echo.

call conda create --prefix "%ENV_PREFIX%" python=3.10 -y
if errorlevel 1 exit /b 1

echo Installing CUDA/cuDNN runtime used by TensorFlow 2.10 on native Windows...
call conda install --prefix "%ENV_PREFIX%" -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
if errorlevel 1 exit /b 1

echo Installing TensorFlow GPU-compatible stack...
call conda run --prefix "%ENV_PREFIX%" python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

call conda run --prefix "%ENV_PREFIX%" python -m pip install "numpy==1.23.5" "tensorflow==2.10.1" "tensorflow-gpu==2.10.1"
if errorlevel 1 exit /b 1

echo Installing core repository dependencies, avoiding newer TensorFlow pins...
call conda run --prefix "%ENV_PREFIX%" python -m pip install ^
    numpy==1.23.5 ^
    scipy==1.10.1 ^
    pandas==2.0.3 ^
    scikit-learn==1.3.0 ^
    matplotlib==3.7.3 ^
    seaborn==0.13.2 ^
    tqdm==4.65.2 ^
    tslearn==0.6.2 ^
    fastdtw==0.3.4 ^
    wildboar==1.1.4 ^
    numba==0.60.0 ^
    llvmlite==0.43.0 ^
    shap==0.42.1 ^
    loguru==0.7.2 ^
    psutil ^
    awkward ^
    sparse
if errorlevel 1 exit /b 1

echo Verifying TensorFlow and GPU visibility...
call conda run --prefix "%ENV_PREFIX%" python -c "import sys, tensorflow as tf; print('python', sys.version); print('tensorflow', tf.__version__); print('gpus', tf.config.list_physical_devices('GPU'))"
if errorlevel 1 exit /b 1

echo Installing CUDA-enabled PyTorch stack for the original MASCOTS MLP surrogate...
call conda run --prefix "%ENV_PREFIX%" python -m pip install ^
    --index-url https://download.pytorch.org/whl/cu124 ^
    "torch==2.4.1+cu124"
if errorlevel 1 exit /b 1

call conda run --prefix "%ENV_PREFIX%" python -m pip install ^
    "pytorch-lightning==2.5.0.post0"
if errorlevel 1 exit /b 1

echo Verifying TensorFlow GPU and Torch CUDA in one process...
call conda run --prefix "%ENV_PREFIX%" python -c "import tensorflow as tf; import torch; print('tf', tf.__version__, tf.config.list_physical_devices('GPU')); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if errorlevel 1 exit /b 1

echo.
echo Done.
echo Next:
echo   conda activate "%ENV_PREFIX%"
