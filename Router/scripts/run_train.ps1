<#
Lightweight PowerShell helper to run LoRA training locally.
Usage examples:
  # Quick run with defaults (uses data\sample.csv and local model)
  .\scripts\run_train.ps1

  # Specify dataset and output dir
  .\scripts\run_train.ps1 -TrainCsv data\train.csv -ValCsv data\val.csv -OutputDir output_lora -Epochs 3 -BatchSize 8

  # Install required Python packages into the activated .venv before running
  .\scripts\run_train.ps1 -InstallDependencies

Notes:
- Ensure repo .venv is present. The script will try to activate .venv\Scripts\Activate.ps1.
- Logs are written to .\logs\ by default and streamed to console.
#>

param(
    [string]$TrainCsv = "data\train.csv",
    [string]$ValCsv = "data\val.csv",
    [string]$ModelDir = "model\chinese-bert-wwm-ext",
    [string]$OutputDir = "output_lora",
    [int]$Epochs = 1,
    [int]$BatchSize = 2,
    [int]$GradientAccumulationSteps = 1,
    [switch]$InstallDependencies,
    [string]$LogFile = ""
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptRoot

# Activate virtualenv if available (repo root .venv)
$venvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtualenv at $venvActivate"
    & $venvActivate
} else {
    Write-Warning ".venv Activate script not found. Make sure you activate your virtualenv manually if needed."
}

# Prefer the venv python executable; fall back to system python
$pythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}
Write-Host "Using Python executable: $pythonExe"

# Install dependencies if requested
if ($InstallDependencies) {
    Write-Host "Installing recommended dependencies into the active Python environment..."
    $pipArgs = @("-m","pip","install","-U","torch","transformers","datasets","peft","accelerate","evaluate")
    & $pythonExe @pipArgs
    if ($LASTEXITCODE -ne 0) { Write-Warning "pip install failed (exit code $LASTEXITCODE)." }
}

# Ensure logs directory
$logsDir = Join-Path $ScriptRoot "logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }
if ([string]::IsNullOrEmpty($LogFile)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $logsDir "train_$timestamp.log"
}

# Build command arguments using absolute script path (do not change current working directory)
$trainScript = Join-Path $RepoRoot "scripts\train_lora.py"
$cmd = @($trainScript,
         "--train_csv", (Join-Path $RepoRoot $TrainCsv),
         "--model_dir", (Join-Path $RepoRoot $ModelDir),
         "--output_dir", (Join-Path $RepoRoot $OutputDir),
         "--epochs", $Epochs.ToString(),
         "--batch_size", $BatchSize.ToString(),
         "--gradient_accumulation_steps", $GradientAccumulationSteps.ToString(),
         "--do_train")
if (-not [string]::IsNullOrEmpty($ValCsv)) { $cmd += @("--val_csv", (Join-Path $RepoRoot $ValCsv), "--do_eval") }

Write-Host "Running: python $($cmd -join ' ')"
Write-Host "Logging to: $LogFile"

# Execute and tee output to log file
try {
    & $pythonExe @cmd 2>&1 | Tee-Object -FilePath $LogFile
} catch {
    Write-Error "Training process failed: $_"
    exit 1
}

Write-Host "Training finished. Log: $LogFile"
