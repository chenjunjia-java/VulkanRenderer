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
  echo [cmake_with_msvc_env] ERROR: VsDevCmd.bat not found. Install VS/Build Tools with C++ toolchain. 1>&2
  exit /b 1
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul

rem ---- Ensure LLVM bin is on PATH (for clang-cl/lld-link/llvm-rc etc.) ----
if exist "C:\Program Files\LLVM\bin\clang-cl.exe" (
  set "PATH=C:\Program Files\LLVM\bin;%PATH%"
)

rem ---- Locate real cmake.exe (avoid recursion if CMake Tools points to this .bat) ----
set "REALCMAKE="

for %%C in (
  "D:\Program Files\CMake\bin\cmake.exe"
  "%ProgramFiles%\CMake\bin\cmake.exe"
  "%ProgramFiles(x86)%\CMake\bin\cmake.exe"
) do (
  if exist %%~C (
    set "REALCMAKE=%%~C"
    goto :found_cmake
  )
)

:found_cmake
if "%REALCMAKE%"=="" (
  rem Fallback to PATH
  set "REALCMAKE=cmake.exe"
)

"%REALCMAKE%" %*
exit /b %ERRORLEVEL%

