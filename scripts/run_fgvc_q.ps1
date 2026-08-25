param(
    [int]$Seed = 1,
    [int]$Steps = 3000,
    [string]$DataRoot = "data",
    [ValidateSet("fgvc", "eurosat", "dtd")]
    [string]$Dataset = "fgvc",
    [ValidateSet("abs", "signed")]
    [string]$QMode = "signed",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $repoRoot) ".venv\Scripts\python.exe"
$metrics = Join-Path $repoRoot "results\${Dataset}_seed$Seed\q_${QMode}_$Steps.csv"
$checkpoint = Join-Path $repoRoot "checkpoints\${Dataset}_q_${QMode}_seed$Seed.pt"
$qOutput = Join-Path $repoRoot "gradient_gate_data\${Dataset}_q_${QMode}_seed$Seed"
New-Item -ItemType Directory -Force (Split-Path -Parent $metrics) | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $checkpoint) | Out-Null

Push-Location $repoRoot
try {
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = if ($previousPythonPath) { "$repoRoot;$previousPythonPath" } else { $repoRoot }
    $arguments = @(
        "experiments/train_stage1.py",
        "--method", "q", "--dataset", $Dataset, "--data_root", $DataRoot,
        "--seed", $Seed, "--shots", 16, "--batch_size", 32,
        "--steps", $Steps, "--lr", "2e-4", "--lr_min", "1e-6",
        "--validation_images_per_group", 16, "--validation_ema_beta", 0.97,
        "--full_validation_every", 200, "--checkpoint_every", 100,
        "--q_init_images", 200, "--q_online_images", 4,
        "--q_ema_beta", 0.95, "--q_mode", $QMode,
        "--checkpoint", $checkpoint, "--metrics_output", $metrics,
        "--q_output", $qOutput
    )
    if ($Resume) { $arguments += "--resume" }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "$Dataset q $QMode seed $Seed failed with exit code $LASTEXITCODE" }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
