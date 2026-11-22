@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Promotion Lens could not start: the project's Python environment is missing.
    echo Expected: %~dp0.venv\Scripts\python.exe
    echo Open this project in Codex to repair its existing environment, then try again.
    pause
    exit /b 1
)

echo Starting Promotion Lens...
echo Open http://127.0.0.1:8501 in your browser.
echo Keep this window open while using the app. Press Ctrl+C to stop it.
echo.

".venv\Scripts\python.exe" -X utf8 -m streamlit run "app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true

if errorlevel 1 (
    echo.
    echo Promotion Lens stopped with an error. Read the message above for details.
    pause
    exit /b 1
)

endlocal
