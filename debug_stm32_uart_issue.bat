@echo off
echo ================================================
echo STM32 Car Damage Detection - Debug Version
echo ================================================

echo.
echo Checking STM32 project structure...
if not exist "RL_STM32_CAR\RL_STM32_CAR.ioc" (
    echo ERROR: STM32 project not found!
    pause
    exit /b 1
)

echo ✅ STM32 project found
echo.

echo ================================================
echo UART DEBUG FIXES APPLIED:
echo ================================================
echo ✅ Fixed UART interrupt callback
echo ✅ Improved JSON parsing for new format  
echo ✅ Added timeout protection for analyzing state
echo ✅ Enhanced error handling and recovery
echo ✅ Added support for decimal confidence values
echo.

echo ================================================  
echo DEBUGGING STEPS:
echo ================================================
echo 1. Upload the updated main.c to STM32
echo 2. Connect UART pins properly:
echo    - STM32 PA9 (TX) → ESP32 RX (GPIO3)
echo    - STM32 PA10 (RX) → ESP32 TX (GPIO1) 
echo    - GND → GND
echo 3. Monitor STM32 behavior on LCD
echo 4. Check ESP32 Serial Monitor for communication
echo.

echo ================================================
echo EXPECTED BEHAVIOR:
echo ================================================
echo STM32 LCD should show:
echo   Line 1: "SCRATCH" (damage type)
echo   Line 2: "48%% | SEV: 0" (confidence)
echo.
echo If still shows "ANALYZING PLEASE WAIT...":
echo   - Check UART connections
echo   - Verify baud rate (115200)
echo   - Check if ESP32 is sending data
echo   - Monitor with oscilloscope/logic analyzer
echo.

echo ================================================
echo TROUBLESHOOTING:
echo ================================================
echo Problem: STM32 stuck at "ANALYZING PLEASE WAIT..."
echo.
echo Possible Causes:
echo 1. UART not receiving data from ESP32
echo 2. JSON parsing failed 
echo 3. Interrupt not working properly
echo 4. Buffer overflow or corruption
echo.
echo Solutions Applied:
echo ✅ Enhanced UART interrupt handling
echo ✅ Improved JSON parser for decimal values
echo ✅ Added timeout protection (10 seconds)
echo ✅ Better error recovery mechanisms
echo.

echo ================================================
echo MANUAL TESTING:
echo ================================================
echo To test UART communication manually:
echo 1. Install: pip install pyserial
echo 2. Run: python debug_uart_stm32.py
echo 3. Select STM32 UART port
echo 4. Send test JSON data
echo 5. Monitor STM32 LCD response
echo.

echo ================================================
echo JSON FORMAT EXPECTED BY STM32:
echo ================================================
echo {
echo   "confidence": 0.4852,
echo   "damage_type": "scratch", 
echo   "status": "damage_detected",
echo   "timestamp": "2025-11-26T21:29:01.345088"
echo }
echo.

echo STM32 will convert confidence 0.4852 → 48.52%%
echo Display: "SCRATCH" on line 1, "49%% | SEV: 0" on line 2
echo.

pause