@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---- Locate VsDevCmd.bat (VS 2022) ----
set "VSDEVCMD="

for %%P in (
  "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
  "D:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
) do (
  if exist %%~P (
    set "VSDEVCMD=%%~P"
    goto :found_vsdevcmd
  )
)

:found_vsdevcmd
if "%VSDEVCMD%"=="" (
  echo [clangd_with_msvc_env] ERROR: VsDevCmd.bat not found. Install VS/Build Tools with C++ toolchain. 1>&2
  exit /b 1
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul

rem ---- Ensure LLVM bin is on PATH (clangd/clang-cl) ----
if exist "C:\Program Files\LLVM\bin\clangd.exe" (
  set "PATH=C:\Program Files\LLVM\bin;%PATH%"
)

rem ---- Run clangd ----
if exist "C:\Program Files\LLVM\bin\clangd.exe" (
  "C:\Program Files\LLVM\bin\clangd.exe" %*
  exit /b %ERRORLEVEL%
)

rem Fallback to PATH
clangd.exe %*
exit /b %ERRORLEVEL%

