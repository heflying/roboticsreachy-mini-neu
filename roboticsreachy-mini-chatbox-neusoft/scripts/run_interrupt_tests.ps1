param(
    [switch]$BargeInTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"

$python = ".\.venv-win\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv-win Python. Run: uv venv --python 3.12 .venv-win && uv sync"
}

& $PSScriptRoot\check_python_env.ps1 -VenvName ".venv-win"

if ($BargeInTests) {
    Write-Host "Running barge-in integration tests..."
    & $python -m pytest -p no:cacheprovider -v `
        tests\cascade\test_interrupt_coordinator.py `
        tests\cascade\test_barge_in_detection.py `
        tests\cascade\test_speech_output_barge_in.py `
        tests\cascade\test_audio_playback_interrupt.py
} else {
    Write-Host "Running all cascade interrupt tests..."
    & $python -m pytest -p no:cacheprovider -v `
        tests\cascade\test_interrupt_coordinator.py `
        tests\cascade\test_barge_in_detection.py `
        tests\cascade\test_speech_output_barge_in.py `
        tests\cascade\test_audio_playback_interrupt.py `
        tests\cascade\test_turn_controller.py `
        tests\cascade\test_sentence_chunker_interrupt.py `
        tests\cascade\test_qwen_tts_cancel.py
}