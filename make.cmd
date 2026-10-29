@echo off
setlocal
REM Cross-machine Windows task runner for TrustFed-SDN.
REM Uses MSYS2/GNU Make when available; otherwise falls back natively to .venv\Scripts\python.exe.

if exist "C:\msys64\mingw64\bin\mingw32-make.exe" (
    "C:\msys64\mingw64\bin\mingw32-make.exe" %*
    exit /b %ERRORLEVEL%
)

where mingw32-make >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    mingw32-make %*
    exit /b %ERRORLEVEL%
)

where make >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    make %*
    exit /b %ERRORLEVEL%
)

set "PY=.venv\Scripts\python.exe"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=help"

if /I "%TARGET%"=="help" (
    echo TrustFed-SDN Windows runner targets ^(native fallback - GNU make not required^):
    echo   install   Install editable package and dev dependencies into .venv
    echo   status    Compare GitHub sync, packages, datasets, and team checklist
    echo   lint      Run ruff and black --check
    echo   format    Format code with black and ruff --fix
    echo   test      Run pytest test suite
    echo   check     Run lint + pytest + package import smoke check
    echo   probe     Run hardware and environment probe script
    exit /b 0
)

if not exist "%PY%" (
    echo [ERROR] Virtualenv Python not found at %PY%. Create .venv first.
    exit /b 1
)

if /I "%TARGET%"=="install" (
    "%PY%" -m pip install -e ".[dev]"
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="status" (
    "%PY%" scripts/team_status.py
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="lint" (
    "%PY%" -m ruff check trustfed scripts tests || exit /b 1
    "%PY%" -m black --check trustfed scripts tests || exit /b 1
    exit /b 0
)
if /I "%TARGET%"=="format" (
    "%PY%" -m black trustfed scripts tests || exit /b 1
    "%PY%" -m ruff check --fix trustfed scripts tests || exit /b 1
    exit /b 0
)
if /I "%TARGET%"=="test" (
    "%PY%" -m pytest
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="check" (
    "%PY%" -m ruff check trustfed scripts tests || exit /b 1
    "%PY%" -m black --check trustfed scripts tests || exit /b 1
    "%PY%" -m pytest || exit /b 1
    "%PY%" -c "import trustfed; print('Import check OK: trustfed', trustfed.__version__)" || exit /b 1
    exit /b 0
)
if /I "%TARGET%"=="probe" (
    "%PY%" scripts/probe_hardware.py
    exit /b %ERRORLEVEL%
)

echo [INFO] Target "%TARGET%" requires GNU make or is scheduled for a later step ^(see Makefile^).
exit /b 1
