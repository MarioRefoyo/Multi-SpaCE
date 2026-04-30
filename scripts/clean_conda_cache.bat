@echo off
setlocal

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)

echo This removes downloaded Conda package archives and unused extracted packages.
echo It does not remove your environments.
echo.

call conda clean --tarballs --packages -y
if errorlevel 1 exit /b 1

echo.
echo Conda package cache cleaned.
