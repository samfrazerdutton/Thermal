# Builds native/ (CUDA kernel lab + C++ runtime) with CMake + Ninja.
#
# Why this script exists: CMake's default Windows generator (Visual Studio)
# cannot compile .cu files on this machine because the CUDA toolkit here has
# no Visual Studio integration installed (see docs/environment-report.md).
# Ninja compiles by invoking nvcc/cl directly, which works once the MSVC
# environment variables are loaded -- verified by hand before writing this
# script (see the Phase 7 commit message for the exact repro).
#
# Usage: powershell -File scripts/build-native.ps1 [-Clean]

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."
$buildDir = Join-Path $repoRoot "build\native"

if ($Clean -and (Test-Path $buildDir)) {
    Remove-Item -Recurse -Force $buildDir
}

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found -- no Visual Studio installation detected. See docs/environment-report.md."
}
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    Write-Error "No Visual Studio installation with the VC++ Tools component was found."
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"

# Load the MSVC environment into THIS process (cmd.exe /c ... && set, captured
# and re-applied -- vcvarsall.bat itself only affects the child cmd process).
$envFile = New-TemporaryFile
cmd.exe /c "`"$vcvars`" x64 >nul 2>&1 && set" > $envFile
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
Remove-Item $envFile

# ninja.exe ships via the pip package (see pyproject.toml [dev] extras / doctor
# check) rather than requiring a separate system install.
$pythonScripts = & python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
if ($pythonScripts -and (Test-Path $pythonScripts)) {
    $env:Path = "$pythonScripts;$env:Path"
}

Write-Host "Configuring with Ninja generator..."
cmake -S $repoRoot -B $buildDir -G Ninja -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building..."
cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built: $buildDir\thermal_kernel_bench.exe"
