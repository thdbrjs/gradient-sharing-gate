param(
    [int]$Seed = 1,
    [int]$Steps = 3000,
    [string]$DataRoot = "data",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $repoRoot) ".venv\Scripts\python.exe"
$metrics = Join-Path $repoRoot "results\fgvc_seed$Seed\raw_$Steps.csv"
$checkpoint = Join-Path $repoRoot "checkpoints\fgvc_raw_seed$Seed.pt"
New-Item -ItemType Directory -Force (Split-Path -Parent $metrics) | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $checkpoint) | Out-Null

Push-Location $repoRoot
try {
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = if ($previousPythonPath) { "$repoRoot;$previousPythonPath" } else { $repoRoot }
    $arguments = @(
        "experiments/train_stage1.py",
        "--method", "raw", "--dataset", "fgvc", "--data_root", $DataRoot,
        "--seed", $Seed, "--shots", 16, "--batch_size", 32,
        "--steps", $Steps, "--lr", "2e-4", "--lr_min", "1e-6",
        "--validation_images_per_group", 16, "--validation_ema_beta", 0.97,
        "--full_validation_every", 200, "--checkpoint_every", 100,
        "--checkpoint", $checkpoint, "--metrics_output", $metrics
    )
    if ($Resume) { $arguments += "--resume" }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "raw seed $Seed failed with exit code $LASTEXITCODE" }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
