@echo off
REM SilenceCut 打包脚本 (无图标版本)
REM 请确保已安装所有依赖: pip install -r requirements.txt

echo ====================================
echo  SilenceCut 打包工具
echo ====================================
echo.

REM 检查 pyinstaller 是否安装
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [错误] PyInstaller 未安装，请先运行:
    echo pip install pyinstaller
    pause
    exit /b 1
)

echo [1/3] 清理旧的构建文件...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
if exist "SilenceCut.spec" del /q SilenceCut.spec

echo [2/3] 开始打包...
pyinstaller --noconfirm --onefile --windowed ^
    --name "SilenceCut" ^
    --hidden-import "pydub" ^
    --hidden-import "pydub.silence" ^
    --hidden-import "scipy.io.wavfile" ^
    --hidden-import "numpy" ^
    silence_cut_gui.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！请检查错误信息。
    pause
    exit /b 1
)

echo.
echo [3/3] 打包完成！
echo.
echo ====================================
echo  输出文件: dist\SilenceCut.exe
echo ====================================
echo.
echo 使用前请确保:
echo   1. 系统已安装 FFmpeg
echo   2. ffmpeg.exe 和 ffprobe.exe 在系统 PATH 中
echo.
echo 测试命令: ffmpeg -version
echo.
pause
