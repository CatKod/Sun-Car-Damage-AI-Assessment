@echo off
echo ================================================
echo STM32 CONFIDENCE DISPLAY - DEBUG VERSION  
echo ================================================

echo.
echo ❌ PROBLEM IDENTIFIED:
echo ================================================
echo Issue: LCD shows "CONF: %" instead of "CONF: 49%"
echo Cause: confidence value is 0 or not parsed correctly
echo.

echo ✅ FIXES APPLIED:
echo ================================================
echo 🔢 Added confidence validation in parsing
echo 🔢 Default fallback values if parsing fails  
echo 🔢 Range checking (0-100%)
echo 🔢 Better format handling for display
echo 🔢 Padding for 16-character line length
echo 🔢 Default last_valid_damage with 100% confidence
echo.

echo ================================================
echo EXPECTED DISPLAY FORMATS:
echo ================================================
echo Case 1 - Damage detected:
echo   Line 1: "    SCRATCH     " 
echo   Line 2: "   CONF: 49%   " (centered)
echo.
echo Case 2 - No damage:
echo   Line 1: "   NO DAMAGE    "
echo   Line 2: "   CONF: 95%   " (centered)
echo.
echo Case 3 - System ready:
echo   Line 1: " SYSTEM READY   "
echo   Line 2: "   CONF: 100%  " (centered)
echo.

echo ================================================
echo CONFIDENCE VALUE FLOW:
echo ================================================
echo 1. ESP32 sends: {"confidence":0.48520970344...}
echo 2. STM32 parses: atof(start) = 0.485209...
echo 3. STM32 converts: 0.485 * 100 = 48.5%
echo 4. STM32 formats: sprintf("CONF: %.0f%%", 48.5) = "CONF: 49%"
echo 5. STM32 displays: "   CONF: 49%   " (with padding)
echo.

echo ================================================  
echo DEBUG STEPS:
echo ================================================
echo 1. Check if confidence_ptr is found in JSON
echo 2. Check if atof() returns valid number
echo 3. Check conversion from decimal to percentage
echo 4. Check sprintf formatting
echo 5. Check LCD_Print output
echo.

echo ================================================
echo VALIDATION ADDED:
echo ================================================
echo ✅ if (conf > 0.0) - ensures valid parsing
echo ✅ if (confidence <= 0.0) - sets default 95%
echo ✅ if (confidence > 100.0) - caps at 100%
echo ✅ display_confidence fallback - ensures display
echo ✅ String padding - ensures 16 characters
echo.

echo ================================================
echo TROUBLESHOOTING:
echo ================================================
echo Still shows "CONF: %"?
echo 1. Check ESP32 JSON format is correct
echo 2. Check UART data received by STM32
echo 3. Check confidence parsing in JSON
echo 4. Monitor sprintf() output values
echo.

echo To debug manually:
echo 1. Use logic analyzer on UART TX/RX
echo 2. Add LCD debug prints for confidence value
echo 3. Check memory values in debugger
echo.

echo ================================================
echo EXPECTED BEHAVIOR AFTER FIX:
echo ================================================
echo ⏰ 00:00 - "SYSTEM READY" / "CONF: 100%"
echo ⏰ 00:15 - "ANALYZING" / "PLEASE WAIT..." (2s)  
echo ⏰ 00:17 - "SCRATCH" / "CONF: 49%" (stable)
echo ⏰ 00:32 - "ANALYZING" / "PLEASE WAIT..." (2s)
echo ⏰ 00:34 - "SCRATCH" / "CONF: 49%" (continues)
echo.

pause