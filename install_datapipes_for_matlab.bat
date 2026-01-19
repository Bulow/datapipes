@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

echo Checking for uv...

:: Check if uv exists
where uv >nul 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv...

    :: Install uv using the official PowerShell installer
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest https://astral.sh/uv/install.ps1 -UseBasicParsing | Invoke-Expression"

    if errorlevel 1 (
        echo ERROR: uv installation script failed.
        exit /b 1
    )

    :: Refresh PATH (User + Machine)
    for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('PATH','User')"`) do set "USERPATH=%%A"
    for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('PATH','Machine')"`) do set "MACHINEPATH=%%A"
    set "PATH=%USERPATH%;%MACHINEPATH%"

    :: Verify uv installation
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv installation failed or uv is not on PATH.
        exit /b 1
    )

    echo uv installed successfully.
) else (
    echo uv is already installed.
)

echo.
echo Installing datapipes using uv...

:: Check if datapipes exists
where datapipes >nul 2>&1
if errorlevel 1 (
    echo Installing datapipes...
    uv tool install --python 3.12 git+https://github.com/Bulow/datapipes
    uv tool upgrade datapipes
    if errorlevel 1 (
        echo ERROR: Failed to install datapipes.
        exit /b 1
    )
) else (
    echo Datapipes is already installed. Updating datapipes...
    uv tool upgrade datapipes
    if errorlevel 1 (
        echo ERROR: Failed to upgrade datapipes.
        exit /b 1
    )
)

uv tool update-shell

:: Refresh PATH again
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('PATH','User')"`) do set "USERPATH=%%A"
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('PATH','Machine')"`) do set "MACHINEPATH=%%A"
set "PATH=%USERPATH%;%MACHINEPATH%"

:: Install matlab integration as a matlab package in the default MATLAB library folder
:: Uses the python environment of the datapipes uv-tool installation
echo.
echo Installing datapipes for MATLAB...
uv tool run datapipes init-matlab
if errorlevel 1 (
    echo ERROR: uv tool run datapipes init-matlab failed.
    exit /b 1
)

echo.
echo Done.
pause
