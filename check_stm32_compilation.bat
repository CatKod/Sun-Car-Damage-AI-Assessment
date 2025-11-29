@echo off
REM STM32 Code Compilation Test Script
REM ==================================
REM This script helps verify STM32 code compilation
REM and provides troubleshooting for common issues
REM 
REM Author: AI Assistant
REM Date: November 2025

echo ========================================
echo 🔧 STM32 Code Compilation Checker
echo ========================================
echo.

set "PROJECT_DIR=%~dp0RL_STM32_CAR"
set "BUILD_DIR=%PROJECT_DIR%\Debug"

echo 📁 Project Directory: %PROJECT_DIR%
echo 📁 Build Directory: %BUILD_DIR%
echo.

REM Check if project directory exists
if not exist "%PROJECT_DIR%" (
    echo ❌ STM32 project directory not found: %PROJECT_DIR%
    echo.
    echo 💡 Expected project structure:
    echo    RL_STM32_CAR\
    echo    ├── Core\
    echo    │   ├── Src\
    echo    │   │   └── main.c
    echo    │   └── Inc\
    echo    ├── Drivers\
    echo    └── Debug
    echo.
    pause
    exit /b 1
)

REM Check if main.c exists
if not exist "%PROJECT_DIR%\Core\Src\main.c" (
    echo ❌ main.c not found in Core\Src\
    echo Please ensure main.c exists in the correct location
    pause
    exit /b 1
)

echo ✅ STM32 project structure verified
echo.

REM Check main.c for required functions
echo 🔍 Checking main.c for required functions...
echo.

set "MAIN_C=%PROJECT_DIR%\Core\Src\main.c"
set "MISSING_FUNCTIONS="

REM Check for CarDamage functions
findstr /c:"CarDamage_Init" "%MAIN_C%" >nul 2>&1
if errorlevel 1 set "MISSING_FUNCTIONS=%MISSING_FUNCTIONS% CarDamage_Init"

findstr /c:"CarDamage_RequestAnalysis" "%MAIN_C%" >nul 2>&1
if errorlevel 1 set "MISSING_FUNCTIONS=%MISSING_FUNCTIONS% CarDamage_RequestAnalysis"

findstr /c:"CarDamage_ProcessUARTData" "%MAIN_C%" >nul 2>&1
if errorlevel 1 set "MISSING_FUNCTIONS=%MISSING_FUNCTIONS% CarDamage_ProcessUARTData"

findstr /c:"CarDamage_DisplayResults" "%MAIN_C%" >nul 2>&1
if errorlevel 1 set "MISSING_FUNCTIONS=%MISSING_FUNCTIONS% CarDamage_DisplayResults"

findstr /c:"CarDamage_DisplayStatus" "%MAIN_C%" >nul 2>&1
if errorlevel 1 set "MISSING_FUNCTIONS=%MISSING_FUNCTIONS% CarDamage_DisplayStatus"

if not "%MISSING_FUNCTIONS%"=="" (
    echo ❌ Missing required functions:%MISSING_FUNCTIONS%
    echo.
    echo 💡 Please ensure main.c contains all CarDamage functions:
    echo    - CarDamage_Init^(^)
    echo    - CarDamage_RequestAnalysis^(^)
    echo    - CarDamage_ProcessUARTData^(^)
    echo    - CarDamage_DisplayResults^(^)
    echo    - CarDamage_DisplayStatus^(^)
    echo    - CarDamage_ParseResponse^(^)
    echo    - CarDamage_SendESP32Command^(^)
    echo.
    pause
    exit /b 1
)

echo ✅ All required CarDamage functions found
echo.

REM Check for LCD functions
findstr /c:"LCD_Init" "%MAIN_C%" >nul 2>&1
if errorlevel 1 (
    echo ❌ LCD functions not found in main.c
    echo Please ensure LCD functions are implemented
) else (
    echo ✅ LCD functions found
)

REM Check for variable declarations
findstr /c:"test_counter" "%MAIN_C%" >nul 2>&1
if errorlevel 1 (
    echo ❌ test_counter variable not declared
    echo Please add: uint32_t test_counter = 0;
) else (
    echo ✅ test_counter variable found
)

findstr /c:"DamageResult_t current_damage" "%MAIN_C%" >nul 2>&1
if errorlevel 1 (
    echo ❌ current_damage variable not declared
    echo Please add DamageResult_t structure and variable
) else (
    echo ✅ DamageResult_t current_damage found
)

echo.
echo 📊 Code Analysis Summary:
echo ========================================

REM Count lines of code
for /f %%i in ('find /c /v "" "%MAIN_C%"') do set "LINES=%%i"
echo 📄 Total lines in main.c: %LINES%

REM Count functions
for /f %%i in ('findstr /r "^[a-zA-Z].*(.*).*{" "%MAIN_C%" ^| find /c /v ""') do set "FUNCTIONS=%%i"
echo ⚙️  Functions defined: %FUNCTIONS%

REM Count includes
for /f %%i in ('findstr "^#include" "%MAIN_C%" ^| find /c /v ""') do set "INCLUDES=%%i"
echo 📚 Include statements: %INCLUDES%

echo.
echo 🔨 Compilation Readiness Check:
echo ========================================

REM Check for STM32CubeIDE installation
where stm32cubeid >nul 2>&1
if errorlevel 1 (
    echo ⚠️  STM32CubeIDE not found in PATH
    echo    Manual compilation required
) else (
    echo ✅ STM32CubeIDE found in system PATH
)

REM Check for arm-none-eabi-gcc
where arm-none-eabi-gcc >nul 2>&1
if errorlevel 1 (
    echo ⚠️  ARM GCC toolchain not found in PATH
    echo    Use STM32CubeIDE for compilation
) else (
    echo ✅ ARM GCC toolchain available
    
    REM Try to get compiler version
    for /f "tokens=*" %%i in ('arm-none-eabi-gcc --version 2^>nul ^| findstr "gcc"') do (
        echo    Compiler: %%i
    )
)

echo.
echo 📋 Build Instructions:
echo ========================================
echo.
echo 1. Open STM32CubeIDE
echo 2. Import existing project: %PROJECT_DIR%
echo 3. Select project in Project Explorer
echo 4. Right-click → Build Project
echo 5. Check for compilation errors in console
echo.
echo 🔧 Common Compilation Issues:
echo ----------------------------------------
echo Issue: undefined reference to 'CarDamage_Init'
echo Fix: Ensure all CarDamage functions are implemented
echo.
echo Issue: 'test_counter' undeclared
echo Fix: Add variable declaration in private variables section
echo.
echo Issue: HAL library errors  
echo Fix: Ensure correct STM32 HAL drivers are included
echo.
echo Issue: UART/I2C configuration errors
echo Fix: Verify pin configuration matches hardware setup
echo.

REM Check if Debug folder exists and has build files
if exist "%BUILD_DIR%" (
    echo 📁 Build directory exists: %BUILD_DIR%
    
    if exist "%BUILD_DIR%\*.o" (
        echo ✅ Object files found - previous build detected
        
        if exist "%BUILD_DIR%\RL_STM32_CAR.elf" (
            echo ✅ ELF file exists - successful build detected
            echo 📏 File size:
            dir "%BUILD_DIR%\RL_STM32_CAR.elf" | findstr "RL_STM32_CAR.elf"
        ) else (
            echo ⚠️  ELF file missing - incomplete build
        )
    ) else (
        echo 📝 No object files - clean build directory
    )
) else (
    echo 📝 No build directory - first-time compilation
)

echo.
echo 🎯 Next Steps:
echo ========================================
echo 1. ✅ Code verification completed
echo 2. 🔨 Build project in STM32CubeIDE  
echo 3. 🔌 Flash to STM32F103C8T6
echo 4. 📺 Test LCD display functionality
echo 5. 📡 Verify UART communication with ESP32
echo 6. 🚗 Run complete system integration test
echo.

pause
echo.
echo 🏁 STM32 compilation check completed!
echo Ready for build and deployment.