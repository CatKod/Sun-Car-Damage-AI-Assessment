@echo off
echo ========================================
echo 🚗 Car Damage AI Detection System
echo ========================================
echo.
echo Choose operation mode:
echo 1. Streamlit Web App (Original)
echo 2. Real-Time Hardware System (NEW)
echo 3. Flask AI Server Only
echo ========================================
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🌐 Starting Streamlit Web App...
    echo Open your browser and go to: http://localhost:8501
    echo.
    streamlit run app/streamlit_app.py --server.port 8501
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting Real-Time Hardware System...
    call run_car_damage_system.bat
) else if "%choice%"=="3" (
    echo.
    echo 🔧 Starting Flask AI Server Only...
    python flask_damage_server.py
) else (
    echo.
    echo ❌ Invalid choice! Please select 1, 2, or 3
    pause
    goto :eof
)

pause
