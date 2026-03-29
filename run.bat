@echo off
title Smart Campus Water AI - Runner
color 0A

echo ===================================================
echo     Smart Campus Water AI - Startup Script
echo ===================================================
echo.

if not exist venv\Scripts\python.exe (
    color 0C
    echo [ERROR] Virtual environment not found.
    echo Please double-click 'install.bat' first!
    pause
    exit /b 1
)

echo [OK] Virtual environment found.
echo.

echo Starting FastAPI Backend on Port 8000...
start "FastAPI Backend" cmd /k "venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo Starting Streamlit Dashboard on Port 8501...
start "Streamlit Dashboard" cmd /k "venv\Scripts\python.exe -m streamlit run dashboard/app.py"

echo.
echo ===================================================
echo   Both services are starting in new windows!
echo ===================================================
echo.
echo   Dashboard : http://localhost:8501
echo   API Docs  : http://localhost:8000/docs
echo.
echo   To STOP: Close the two terminal windows that opened.
echo ===================================================
echo.
pause
