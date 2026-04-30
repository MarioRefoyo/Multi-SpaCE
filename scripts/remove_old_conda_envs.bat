@echo off
setlocal

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

echo This will remove the following Conda environments:
echo   counterfactuals
echo   tsinterpret
echo   rt
echo   quant
echo   qfin
echo   AlgorithmicTradingPlatform
echo   qfin3.12.8
echo   cfe_rl
echo   counterfactuals_stumpy
echo.
echo Environments not listed above will not be removed.
echo.

set /p CONFIRM=Type DELETE to continue: 
if not "%CONFIRM%"=="DELETE" (
    echo Aborted.
    exit /b 1
)

for %%E in (
    counterfactuals
    tsinterpret
    rt
    quant
    qfin
    AlgorithmicTradingPlatform
    qfin3.12.8
    cfe_rl
    counterfactuals_stumpy
) do (
    echo.
    echo Removing %%E...
    call conda env remove --name "%%E" -y
    if errorlevel 1 (
        echo WARNING: Failed to remove %%E. Continuing with the next environment.
    )
)

echo.
echo Cleaning Conda package cache...
call conda clean --tarballs --packages -y

echo.
echo Done.
