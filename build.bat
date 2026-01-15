@echo off
chcp 65001 >nul
echo ========================================
echo   SilenceCut - PyInstaller Build Script
echo ========================================
echo.

REM Check if pyinstaller is installed
where pyinstaller >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller not found!
    echo Please install: pip install pyinstaller
    pause
    exit /b 1
)

REM Check icon file
set ICON_OPT=

if exist "icon1.ico" (
    set ICON_OPT=--icon=icon1.ico
    echo [INFO] Using icon: icon1.ico
) else if exist "icon2.ico" (
    set ICON_OPT=--icon=icon2.ico
    echo [INFO] Using icon: icon2.ico
) else if exist "icon3.ico" (
    set ICON_OPT=--icon=icon3.ico
    echo [INFO] Using icon: icon3.ico
) else (
    echo [INFO] No .ico file found, building without icon
    echo [TIP] Convert PNG to ICO and name it icon1.ico, icon2.ico, or icon3.ico
)

echo.
echo [1/3] Cleaning old build files...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "*.spec" del /q *.spec

echo [2/3] Building executable...
pyinstaller --onefile --windowed --name SilenceCut --clean %ICON_OPT% silence_cut_gui.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [3/3] Build complete!
echo.
echo Output: dist\SilenceCut.exe
echo.

if exist "dist\SilenceCut.exe" (
    echo Opening output folder...
    explorer dist
)

pause
