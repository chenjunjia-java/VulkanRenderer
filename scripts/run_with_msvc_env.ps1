param(
    [Parameter(Mandatory = $true)]
    [string]$Command,

    [ValidateSet("x64", "x86", "arm64")]
    [string]$Arch = "x64",

    [ValidateSet("x64", "x86", "arm64")]
    [string]$HostArch = "x64"
)

$vsDevCmdCandidates = @(
    "$env:ProgramFiles\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat",
    "D:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
    "D:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
)

$vsDevCmd = $vsDevCmdCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $vsDevCmd) {
    throw "VsDevCmd.bat not found. Install Visual Studio (or Build Tools) with 'Desktop development with C++' / MSVC toolchain."
}

Write-Host "Using: $vsDevCmd"

# Ensure clang-cl is available even if LLVM isn't on PATH yet (common for winget installs).
$llvmBinCandidates = @(
    "C:\Program Files\LLVM\bin",
    "C:\LLVM\bin"
)

$llvmBin = $llvmBinCandidates | Where-Object { Test-Path (Join-Path $_ "clang-cl.exe") } | Select-Object -First 1
if ($llvmBin) {
    Write-Host "Using LLVM bin: $llvmBin"
}

$llvmPathCmd = ""
if ($llvmBin) {
    # Use cmd.exe syntax to prepend PATH inside the VsDevCmd environment.
    $llvmPathCmd = "set ""PATH=$llvmBin;%PATH%"" >nul && "
}

cmd.exe /c "`"$vsDevCmd`" -arch=$Arch -host_arch=$HostArch >nul && $llvmPathCmd $Command"
$code = $LASTEXITCODE

# Per clangd docs, it will auto-discover compile_commands.json in the source root or in a "build/" dir.
# Our build dir is build/clang-cl-debug/, so copy a convenience copy into build/ for auto-discovery.
if ($code -eq 0) {
    try {
        $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
        $from = Join-Path $repoRoot "build\clang-cl-debug\compile_commands.json"
        $toDir = Join-Path $repoRoot "build"
        $to = Join-Path $toDir "compile_commands.json"

        if (Test-Path $from) {
            if (!(Test-Path $toDir)) { New-Item -ItemType Directory -Path $toDir | Out-Null }
            Copy-Item -Path $from -Destination $to -Force
        }
    } catch {
        # Non-fatal: build/configure succeeded; copying is best-effort.
    }
}

exit $code

