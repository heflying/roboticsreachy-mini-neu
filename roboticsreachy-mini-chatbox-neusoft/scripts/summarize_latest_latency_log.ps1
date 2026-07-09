param(
    [string]$LogDirectory = "logs"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$latest = Get-ChildItem -Path $LogDirectory -Filter "chatbox-*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latest) {
    throw "No chatbox log found under $LogDirectory."
}

Write-Host "Latest log: $($latest.FullName)"
Write-Host ""
Write-Host "Mode markers:"
Select-String -Path $latest.FullName -Pattern `
    "Cascade: ASR=|streaming dialog mode|dialog-only tool mode|tts_ws_reused|tts_ws_prepared_stale|asr_ws_reused|llm_first_speech_chunk|streaming_dialog_first_audio" |
    Select-Object -Last 30 |
    ForEach-Object { $_.Line }

Write-Host ""
Write-Host "Latency summaries:"
Select-String -Path $latest.FullName -Pattern `
    "ASR Processing|LLM Generation|TTS time to first audio|TTS request count|Speech End .+ First Tool|Speech End .+ First Audio|Speech End .+ Playback Done|Speech End .+ Audio Queued|PERCEIVED: Speech End -> VAD Ready|ASR WS|ASR cloud final wait|TTS WS|TTS cloud first audio wait|TTS stream after first chunk|TTS queue all audio|tts_audio_chunk_gap|tts_ws_prepared_stale|tts_ws_reuse_failed|Timed out waiting for Qwen realtime TTS first audio" |
    Select-Object -Last 80 |
    ForEach-Object { $_.Line }
