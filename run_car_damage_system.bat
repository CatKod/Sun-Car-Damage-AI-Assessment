@echo off
REM Real-Time Car Damage Detection System Startup Script
REM ====================================================
REM This script starts the Flask AI server for ESP32-CAM integration
REM Author: AI Assistant
REM Date: November 2025

echo ========================================
echo 🚗 Real-Time Car Damage Detection System
echo ========================================
echo.
echo Starting Flask AI Server for ESP32-CAM...
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Get current directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo 📦 Checking required packages...
echo.

REM Install required packages if not present
pip show flask >nul 2>&1 || (
    echo Installing Flask...
    pip install flask
)

pip show ultralytics >nul 2>&1 || (
    echo Installing Ultralytics YOLO...
    pip install ultralytics
)

pip show opencv-python >nul 2>&1 || (
    echo Installing OpenCV...
    pip install opencv-python
)

pip show pillow >nul 2>&1 || (
    echo Installing Pillow...
    pip install pillow
)

pip show numpy >nul 2>&1 || (
    echo Installing NumPy...
    pip install numpy
)

echo.
echo ✅ All packages ready!
echo.

REM Get local IP address for ESP32 configuration
echo 🌐 Network Configuration:
echo ========================================
echo Your PC IP addresses:

REM Try to get WiFi adapter IP (most common for ESP32 connection)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set ip=%%a
    set ip=!ip: =!
    if not "!ip!"=="127.0.0.1" (
        echo   • !ip!
    )
)

echo.
echo 📋 ESP32-CAM Configuration Steps:
echo ========================================
echo 1. Update ESP32 code with your WiFi credentials:
echo    - const char* ssid = "YOUR_WIFI_NAME";
echo    - const char* password = "YOUR_WIFI_PASSWORD";
echo.
echo 2. Update Flask server IP in ESP32 code:
echo    - const char* flask_server = "http://YOUR_PC_IP:5000";
echo.
echo 3. Upload ESP32_Car_Damage_Bridge.ino to ESP32-CAM
echo 4. Upload updated main.c to STM32F103C8T6
echo 5. Connect hardware as per wiring diagram
echo.

echo 🚀 Starting Flask AI Server...
echo ========================================
echo Server will be available at:
echo   - Local: http://localhost:5000
echo   - Network: http://YOUR_PC_IP:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start Flask server
python flask_damage_server.py

echo.
echo 👋 Server stopped.
pause