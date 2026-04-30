@echo off
setlocal

set "ENV_NAME=counterfactuals_tf_gpu"
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "WORK_DRIVE=%~d0"
set "CONDA_PKGS_DIRS=%WORK_DRIVE%\.conda_pkgs"
set "PIP_CACHE_DIR=%WORK_DRIVE%\.pip_cache"
set "TMP=%WORK_DRIVE%\.tmp"
set "TEMP=%WORK_DRIVE%\.tmp"

if not exist "%CONDA_PKGS_DIRS%" mkdir "%CONDA_PKGS_DIRS%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

echo Creating environment: %ENV_NAME%
echo Conda package cache: %CONDA_PKGS_DIRS%
echo Pip cache: %PIP_CACHE_DIR%
echo.

call conda create --name "%ENV_NAME%" python=3.8.18 -y
if errorlevel 1 exit /b 1

echo Installing CUDA/cuDNN runtime for native Windows TensorFlow GPU...
call conda install --name "%ENV_NAME%" -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
if errorlevel 1 exit /b 1

echo Installing TensorFlow GPU-compatible stack...
call conda run --name "%ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

call conda run --name "%ENV_NAME%" python -m pip install ^
    "numpy==1.24.3" ^
    "tensorflow==2.10.1" ^
    "tensorflow-gpu==2.10.1" ^
    "keras==2.10.0" ^
    "protobuf==3.19.6" ^
    "h5py==3.7.0"
if errorlevel 1 exit /b 1

echo Installing repository dependencies used by the original counterfactual experiments...
call conda run --name "%ENV_NAME%" python -m pip install ^
    alibi==0.9.4 ^
    dask==2023.5.0 ^
    distributed==2023.5.0 ^
    fastdtw==0.3.4 ^
    matplotlib==3.7.3 ^
    matrixprofile==1.1.10 ^
    numba==0.58.0 ^
    pandas==2.0.3 ^
    plotly==5.24.1 ^
    pytest==8.3.5 ^
    ray==2.53.0 ^
    requests==2.31.0 ^
    scikit-learn==1.3.0 ^
    scikit-optimize==0.9.0 ^
    scipy==1.10.1 ^
    seaborn==0.13.0 ^
    shap==0.42.1 ^
    stumpy==1.13.0 ^
    torchsummary==1.5.1 ^
    tqdm==4.65.2 ^
    TSInterpret==0.4.6 ^
    tslearn==0.6.2 ^
    wildboar==1.1.4
if errorlevel 1 exit /b 1

echo Installing CUDA-enabled PyTorch used by previous experiments...
call conda run --name "%ENV_NAME%" python -m pip install ^
    --index-url https://download.pytorch.org/whl/cu124 ^
    "torch==2.4.1+cu124"
if errorlevel 1 exit /b 1

echo Verifying TensorFlow GPU and Torch CUDA in one process...
call conda run --name "%ENV_NAME%" python -c "import tensorflow as tf; import torch; print('tf', tf.__version__, tf.config.list_physical_devices('GPU')); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if errorlevel 1 exit /b 1

echo.
echo Done.
echo Next:
echo   conda activate %ENV_NAME%
