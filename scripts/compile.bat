@echo off
setlocal

rem Prefer Vulkan SDK environment variable if present.
if defined VULKAN_SDK (
    set "GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
) else (
    rem Fallback to a common default install path (adjust if needed).
    set "GLSLC=D:\VulkanSDK\1.4.335.0\Bin\glslc.exe"
)
set "SHADER_DIR=%~dp0..\assets\shaders"

if not exist "%GLSLC%" (
    echo [Error] glslc not found: "%GLSLC%"
    echo         Please install Vulkan SDK or set VULKAN_SDK env var.
    exit /b 1
)

echo Compiling shaders in "%SHADER_DIR%" using "%GLSLC%"...

"%GLSLC%" "%SHADER_DIR%\VertShaders\shader.vert" -o "%SHADER_DIR%\VertShaders\shader.vert.spv"
if errorlevel 1 exit /b 1
"%GLSLC%" "%SHADER_DIR%\FragShaders\shader.frag" -o "%SHADER_DIR%\FragShaders\shader.frag.spv"
if errorlevel 1 exit /b 1

"%GLSLC%" --target-env=vulkan1.2 "%SHADER_DIR%\VertShaders\pbr.vert" -o "%SHADER_DIR%\VertShaders\pbr.vert.spv"
if errorlevel 1 exit /b 1
"%GLSLC%" --target-env=vulkan1.2 "%SHADER_DIR%\FragShaders\pbr.frag" -o "%SHADER_DIR%\FragShaders\pbr.frag.spv"
if errorlevel 1 exit /b 1

echo Done.
exit /b 0