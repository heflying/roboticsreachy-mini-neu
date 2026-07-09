param(
    [switch]$DebugLog,
    [switch]$NoStreamingDialog,
    [switch]$AllowActionTools,
    [string]$LogDirectory = "logs",
    [string]$LlmProvider = "qwen-flash",
    [string]$AsrProvider = "qwen_realtime_asr",
    [string]$TtsProvider = "qwen_realtime_tts",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
$env:CASCADE_STREAMING_DIALOG = if ($NoStreamingDialog) { "0" } else { "1" }
$env:CASCADE_DIALOG_ONLY_TOOLS = if ($AllowActionTools) { "0" } else { "1" }

$python = ".\.venv-win\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv-win Python. Run: uv venv --python 3.12 .venv-win && uv sync"
}

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $LogDirectory "chatbox-$timestamp.log"
$latestLogPath = Join-Path $LogDirectory "latest.log"

$appArgs = @(
    "-m", "reachy_mini_conversation_app.main",
    "--gradio",
    "--no-camera",
    "--asr-provider", $AsrProvider,
    "--llm-provider", $LlmProvider,
    "--tts-provider", $TtsProvider
)

if ($DebugLog) {
    $appArgs += "--debug"
}

Write-Host "Log file: $logPath"
Write-Host "Latest log: $latestLogPath (for Claude Code analysis)"
Write-Host "CASCADE_STREAMING_DIALOG=$env:CASCADE_STREAMING_DIALOG"
Write-Host "CASCADE_DIALOG_ONLY_TOOLS=$env:CASCADE_DIALOG_ONLY_TOOLS"
Write-Host "Command: $python $($appArgs -join ' ')"

if ($DryRun) {
    return
}

& $PSScriptRoot\check_python_env.ps1

$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    # Clear latest.log for fresh session
    if (Test-Path $latestLogPath) {
        Remove-Item $latestLogPath -Force
    }

    # Run app, tee to timestamped log and also write to latest.log
    & $python @appArgs 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $logPath |
        Out-File -FilePath $latestLogPath -Encoding utf8 -Append
}
finally {
    $ErrorActionPreference = $oldErrorActionPreference
}
