@echo off
setlocal

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

if not exist "environment_counterfactuals_tf_gpu.yml" (
    echo ERROR: environment_counterfactuals_tf_gpu.yml not found.
    echo Run this script from the repository root.
    exit /b 1
)

echo Creating counterfactuals_tf_gpu from exported environment file.
echo This is the most faithful option, but can be less portable than
echo scripts\create_counterfactuals_tf_gpu_env.bat.
echo.

call conda env create -f environment_counterfactuals_tf_gpu.yml
if errorlevel 1 exit /b 1

echo.
echo Verifying TensorFlow GPU and Torch CUDA...
call conda run --name counterfactuals_tf_gpu python -c "import tensorflow as tf; import torch; print('tf', tf.__version__, tf.config.list_physical_devices('GPU')); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if errorlevel 1 exit /b 1

echo.
echo Done.
