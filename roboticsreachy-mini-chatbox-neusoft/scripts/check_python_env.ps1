param(
    [string]$ExpectedMinor = "3.12",
    [string]$VenvName = ".venv-win"  # Windows使用.venv-win，WSL使用.venv
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

Write-Host "Repo: $repoRoot"
Write-Host "UV_CACHE_DIR: $env:UV_CACHE_DIR"

if (-not (Test-Path ".\.python-version")) {
    throw "Missing .python-version. Expected $ExpectedMinor."
}

$pinned = (Get-Content ".\.python-version" -Raw).Trim()
Write-Host ".python-version: $pinned"
if (-not $pinned.StartsWith($ExpectedMinor)) {
    throw ".python-version is '$pinned', expected $ExpectedMinor."
}

$venvPython = ".\$VenvName\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Missing $VenvName. Create it with: uv venv --python $ExpectedMinor $VenvName"
}
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$version = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>&1
$versionExitCode = $LASTEXITCODE
$ErrorActionPreference = $oldErrorActionPreference
if ($versionExitCode -ne 0) {
    throw "$VenvName Python failed to run. Output: $version. Recreate the venv with: uv venv --python $ExpectedMinor $VenvName"
}
Write-Host "$VenvName Python: $version"
if (-not $version.StartsWith($ExpectedMinor)) {
    throw "$VenvName uses Python $version, expected $ExpectedMinor. Recreate the venv."
}

$ErrorActionPreference = "Continue"
$prefix = & $venvPython -c "import sys; print(sys.prefix)" 2>&1
$prefixExitCode = $LASTEXITCODE
$ErrorActionPreference = $oldErrorActionPreference
if ($prefixExitCode -ne 0) {
    throw "Failed to inspect $VenvName sys.prefix. Output: $prefix"
}
Write-Host "sys.prefix: $prefix"

Write-Host "OK: Windows Python environment ($VenvName) is pinned to $ExpectedMinor."
