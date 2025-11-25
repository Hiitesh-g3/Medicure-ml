@echo off
title MediCure ML Service
echo =========================================
echo     Starting MediCure ML Microservice
echo =========================================

REM Change directory to script folder
cd /d %~dp0

REM ---------------------------
REM Activate Virtual Environment
REM ---------------------------
echo Activating virtual environment...
call venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    echo Make sure venv exists: python -m venv venv
    pause
    exit /b
)

REM ---------------------------
REM Set Environment Variables
REM ---------------------------
echo Setting environment variables...

set GEMINI_API_KEY=AIzaSyCglOVm9tx5I2Er2zPfMDQkWHEPen3kdik
set PATH=%PATH%;C:\Program Files\Tesseract-OCR\tesseract.exe

REM ---------------------------
REM Start Uvicorn ML API
REM ---------------------------
echo Starting ML Service on port 9000...
echo =========================================

uvicorn ml_api:app --host 0.0.0.0 --port 9000 --reload

echo -----------------------------------------
echo Server stopped or crashed.
pause
