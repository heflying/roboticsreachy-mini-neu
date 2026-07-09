param(
    [switch]$AllCascade
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

$python = ".\.venv-win\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv-win Python. Run: uv venv --python 3.12 .venv-win && uv sync --active"
}

& $PSScriptRoot\check_python_env.ps1 -VenvName ".venv-win"

if ($AllCascade) {
    & $python -m pytest -p no:cacheprovider tests\cascade
} else {
    & $python -m pytest -p no:cacheprovider `
        tests\cascade\test_streaming_dialog.py `
        tests\cascade\test_qwen_realtime_asr.py `
        tests\cascade\test_qwen_config.py `
        tests\cascade\test_qwen_llm.py `
        tests\cascade\test_qwen_realtime_tts.py
}
