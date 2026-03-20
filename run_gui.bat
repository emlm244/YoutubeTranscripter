@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   YouTube Transcriber - Smart Launcher
echo ========================================
echo.

REM ============================================
REM Step 1: Check Python availability
REM ============================================
echo [1/5] Checking Python installation...

REM Check if Python 3.12 is available (preferred for CUDA support)
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    REM Fall back to Python 3.11
    py -3.11 --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python 3.11 or 3.12 is required for GPU acceleration
        echo.
        echo Current issue: PyTorch with CUDA support is NOT available for Python 3.13+
        echo.
        echo Please install Python 3.12 from:
        echo   - Official: https://www.python.org/downloads/
        echo   - Or run: winget install Python.Python.3.12
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=py -3.11"
    set "PYTHON_VERSION=3.11"
) else (
    set "PYTHON_CMD=py -3.12"
    set "PYTHON_VERSION=3.12"
)

REM Get exact Python version for display
for /f "tokens=2" %%V in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_FULL_VERSION=%%V"
echo [OK] Python %PYTHON_FULL_VERSION% found
echo.

REM ============================================
REM Step 2: Virtual Environment Setup
REM ============================================
echo [2/5] Checking virtual environment...
set "FIRST_RUN=0"
set "REBUILD_VENV=0"

if exist "venv\" (
    if not exist "venv\Scripts\python.exe" set "REBUILD_VENV=1"
    if not exist "venv\Scripts\activate.bat" set "REBUILD_VENV=1"
    if not exist "venv\pyvenv.cfg" set "REBUILD_VENV=1"

    if "!REBUILD_VENV!"=="0" (
        set "VENV_VERSION="
        for /f %%V in ('venv\Scripts\python.exe -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))" 2^>nul') do set "VENV_VERSION=%%V"
        if not defined VENV_VERSION set "REBUILD_VENV=1"
    )

    if "!REBUILD_VENV!"=="0" if /I not "!VENV_VERSION!"=="%PYTHON_VERSION%" (
        echo [INFO] Existing virtual environment uses Python !VENV_VERSION!, expected %PYTHON_VERSION%.
        set "REBUILD_VENV=1"
    )

    if "!REBUILD_VENV!"=="1" (
        echo [INFO] Existing virtual environment is machine-specific, incomplete, or uses the wrong Python version.
        echo [INFO] Rebuilding it for this machine...
        rmdir /s /q venv
        if exist "venv\" (
            echo [ERROR] Failed to remove the old virtual environment
            echo.
            echo Close any open terminals or editors using this folder, then try again.
            echo.
            pause
            exit /b 1
        )
    )
)

if not exist "venv\" (
    echo Creating virtual environment with Python %PYTHON_VERSION%...
    echo This is a one-time setup and may take a minute.
    echo.
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully with Python %PYTHON_VERSION%
    set "FIRST_RUN=1"
) else (
    echo [OK] Compatible virtual environment found
)
echo.

REM ============================================
REM Step 3: Activate Virtual Environment
REM ============================================
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM ============================================
REM Step 4: Dependency Management
REM ============================================
echo [4/5] Checking install prerequisites and Python packages...

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo [WARNING] requirements.txt not found
    echo Skipping dependency check...
    echo.
    goto skip_dependencies
)

findstr /i /c:"git+" requirements.txt >nul 2>&1
if not errorlevel 1 (
    git --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git is required because requirements.txt contains git-based dependencies.
        echo.
        echo Please install Git from:
        echo   - Official: https://git-scm.com/download/win
        echo   - Or run: winget install --id Git.Git -e
        echo.
        pause
        exit /b 1
    )
    echo [OK] Git found for git-based Python dependencies
) else (
    echo [OK] No git-based Python dependencies detected
)

echo [INFO] First-time setup requires internet access for package and model downloads.
echo.

REM Always check dependencies, but be smart about it
if !FIRST_RUN!==1 (
    echo Installing dependencies from requirements.txt...
    echo This may take a few minutes on first run.
    echo.
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo.
        echo Common causes:
        echo   - No internet connection
        echo   - Git not installed for git-based packages
        echo   - Python package index temporarily unavailable
        echo.
        pause
        exit /b 1
    )
    echo [OK] All dependencies installed
) else (
    REM Quick check - validate the same startup path the GUI uses.
    python -c "import gui_runtime_bootstrap, youtube_transcript_api, faster_whisper, yt_dlp, sounddevice, language_tool_python, transformers, gector, PyQt6" >nul 2>&1
    if errorlevel 1 (
        echo Dependencies missing or outdated. Installing...
        pip install -q -r requirements.txt
        if errorlevel 1 (
            echo [ERROR] Failed to install dependencies
            echo.
            echo Common causes:
            echo   - No internet connection
            echo   - Git not installed for git-based packages
            echo   - Python package index temporarily unavailable
            echo.
            pause
            exit /b 1
        )

        python -c "import gui_runtime_bootstrap, youtube_transcript_api, faster_whisper, yt_dlp, sounddevice, language_tool_python, transformers, gector, PyQt6" >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Dependencies still failed validation after installation
            echo.
            pause
            exit /b 1
        )

        echo [OK] Dependencies updated
    ) else (
        echo [OK] All dependencies satisfied
    )
)
echo.

:skip_dependencies

REM ============================================
REM Step 5: Runtime Prerequisites
REM ============================================
echo [5/5] Checking runtime prerequisites...

set "FFMPEG_LOCATION="
for /f "usebackq delims=" %%F in (`python -c "from youtube_transcriber import find_ffmpeg; print(find_ffmpeg() or '')" 2^>nul`) do set "FFMPEG_LOCATION=%%F"

if defined FFMPEG_LOCATION (
    echo [OK] FFmpeg found at !FFMPEG_LOCATION!
) else (
    where ffmpeg >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] FFmpeg not found in PATH or common install locations
        echo FFmpeg is required for audio processing.
        echo.

        where winget >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] winget is not available, so FFmpeg cannot be auto-installed here.
            echo Please install FFmpeg manually:
            echo   - winget install FFmpeg
            echo   - choco install ffmpeg
            echo   - https://ffmpeg.org/download.html
            echo.
        ) else (
            echo Attempting auto-installation via winget...
            winget install --id=Gyan.FFmpeg -e --silent --accept-source-agreements --accept-package-agreements >nul 2>&1
            if errorlevel 1 (
                echo [WARNING] Auto-install failed. Please install FFmpeg manually:
                echo.
                echo Option 1: winget install FFmpeg
                echo Option 2: choco install ffmpeg
                echo Option 3: Download from https://ffmpeg.org/download.html
                echo.
                echo The application will still start, but audio processing may fail.
                echo.
            ) else (
                echo [OK] FFmpeg install command completed
                echo Note: PATH updates may require a new terminal session.
                echo.
            )
        )
    ) else (
        echo [OK] FFmpeg found in PATH
    )
)
echo.

echo Checking first-run AI model downloads...
python launcher_preflight.py || echo [WARNING] Could not determine model cache status. The application can still continue.
echo.

echo Checking GPU acceleration support...
python -c "from torch_runtime import get_torch; from youtube_transcriber import get_whisper_cuda_status; torch = get_torch(context='run_gui:gpu_check'); fw_ok, fw_name = get_whisper_cuda_status(); print('[OK] PyTorch with CUDA:', bool(torch and torch.cuda.is_available())); print('[OK] faster-whisper CUDA backend:', fw_ok, fw_name if fw_name else '')" 2>nul || echo [INFO] GPU check failed - application will auto-fallback to CPU mode
echo.

REM ============================================
REM Launch Application
REM ============================================
echo ========================================
echo   Launching YouTube Transcriber GUI
echo ========================================
echo.
echo Logs will be saved to:
echo   - gui_transcriber.log
echo   - youtube_transcriber.log
echo.

python gui_transcriber.py

REM ============================================
REM Error Handling
REM ============================================
if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Application crashed
    echo ========================================
    echo.
    echo Please check the log files for details:
    echo   - gui_transcriber.log
    echo   - youtube_transcriber.log
    echo.
    echo Common issues:
    echo   - Missing dependencies: Delete 'venv' folder and run again
    echo   - Missing Git: Install Git before dependency setup
    echo   - FFmpeg not found: Install FFmpeg manually
    echo   - First-run model downloads blocked: Check internet/firewall access
    echo   - CUDA errors: Application will use CPU mode
    echo.
)

echo.
pause
