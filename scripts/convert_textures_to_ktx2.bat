@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM Use `toktx` from the system PATH (make sure the toktx directory is added to PATH).
where toktx >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 toktx，请确保已将其所在目录加入系统 PATH 环境变量。
    exit /b 1
)

set /p "TEXTURES_DIR=请输入需要转换的文件夹路径: "
set /p "OUTPUT_DIR=请输入输出文件夹路径: "

REM Strip leading/trailing quotes and spaces from paths.
for /f "tokens=*" %%a in ("%TEXTURES_DIR%") do set "TEXTURES_DIR=%%~a"
for /f "tokens=*" %%a in ("%OUTPUT_DIR%") do set "OUTPUT_DIR=%%~a"

if not exist "%TEXTURES_DIR%" (
    echo [错误] 未找到输入文件夹: %TEXTURES_DIR%
    pause
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo 正在将贴图转换为 KTX2 格式...
echo 输出目录: %OUTPUT_DIR%
echo.

REM Supported input formats: png, jpg
set "COUNT=0"
for %%F in ("%TEXTURES_DIR%\*.png" "%TEXTURES_DIR%\*.jpg") do (
    set "INPUT=%%F"
    set "BASENAME=%%~nF"
    set "OUTPUT=%OUTPUT_DIR%\!BASENAME!.ktx2"

    echo 转换: %%~nxF -^> !BASENAME!.ktx2
    REM Notes: --t2 outputs KTX2; --genmipmap generates mipmaps.
    REM `--bcmp` is deprecated; use `--encode etc1s` instead. Vulkan convention: --lower_left_maps_to_s0t0
    toktx --t2 --genmipmap --encode etc1s --lower_left_maps_to_s0t0 --assign_oetf srgb "!OUTPUT!" "!INPUT!"
    if !errorlevel! equ 0 (
        set /a COUNT+=1
    ) else (
        echo [警告] 转换失败: %%~nxF
    )
)

echo.
echo 完成，共成功转换 %COUNT% 个文件。
pause
