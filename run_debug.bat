@echo off
REM 以正确的工作目录运行 VulkanLearning（项目根目录才能找到 assets/）
cd /d "%~dp0"
"build\Debug\VulkanLearning.exe"
if errorlevel 1 (
    echo.
    echo 程序异常退出，按任意键关闭...
    pause >nul
)
