@echo off
REM 以正确的工作目录运行 VulkanLearning Release 版本
cd /d "%~dp0"
"build\Release\VulkanLearning.exe"
if errorlevel 1 (
    echo.
    echo 程序异常退出，按任意键关闭...
    pause >nul
)
