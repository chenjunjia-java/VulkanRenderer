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

rem Legacy simple shaders (kept for old samples)
"%GLSLC%" --target-env=vulkan1.2 "%SHADER_DIR%\VertShaders\shader.vert" -o "%SHADER_DIR%\VertShaders\shader.vert.spv"
if errorlevel 1 exit /b 1
"%GLSLC%" --target-env=vulkan1.2 "%SHADER_DIR%\FragShaders\shader.frag" -o "%SHADER_DIR%\FragShaders\shader.frag.spv"
if errorlevel 1 exit /b 1

rem Main engine shaders (keep aligned with CMakeLists.txt SHADER_SOURCES)
set SHADERS=^
VertShaders\pbr.vert ^
FragShaders\pbr.frag ^
VertShaders\depth_prepass.vert ^
FragShaders\depth_only.frag ^
VertShaders\occlusion_bounds.vert ^
VertShaders\cubemap_capture.vert ^
FragShaders\equirect_to_cubemap.frag ^
VertShaders\skybox.vert ^
FragShaders\skybox.frag ^
FragShaders\irradiance_convolution.frag ^
VertShaders\prefilter_capture.vert ^
FragShaders\prefilter.frag ^
VertShaders\brdf_quad.vert ^
FragShaders\brdf_integrate.frag ^
VertShaders\fullscreen.vert ^
FragShaders\bloom_extract.frag ^
FragShaders\bloom_blur.frag ^
FragShaders\tonemap_bloom.frag ^
CompShaders\rtao_trace_half.comp ^
CompShaders\rtao_atrous.comp ^
CompShaders\rtao_upsample.comp

for %%F in (%SHADERS%) do (
    "%GLSLC%" --target-env=vulkan1.2 "%SHADER_DIR%\%%F" -o "%SHADER_DIR%\%%F.spv"
    if errorlevel 1 exit /b 1
)

echo Done.
exit /b 0