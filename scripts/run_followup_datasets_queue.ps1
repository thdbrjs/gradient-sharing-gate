param(
    [int]$FirstSeed = 1,
    [int]$LastSeed = 5,
    [int]$Steps = 3000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path (Split-Path -Parent $repoRoot) "data"
$log = Join-Path $repoRoot "results\followup_datasets_queue.log"

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -LiteralPath $log -Value $line
    Write-Output $line
}

function Last-Step([string]$Metrics) {
    if (-not (Test-Path -LiteralPath $Metrics)) { return 0 }
    $last = Get-Content -LiteralPath $Metrics -Tail 1
    if (-not $last -or $last.StartsWith("method,")) { return 0 }
    return [int](($last -split ',')[2])
}

function Run-One([string]$Dataset, [int]$Seed, [string]$Mode) {
    $prefix = Join-Path $repoRoot "results\${Dataset}_seed$Seed"
    $metrics = if ($Mode -eq "raw") {
        Join-Path $prefix "raw_$Steps.csv"
    } else {
        Join-Path $prefix "q_${Mode}_$Steps.csv"
    }
    if ((Last-Step $metrics) -ge $Steps) {
        Write-QueueLog "skip completed dataset=$Dataset mode=$Mode seed=$Seed"
        return
    }
    $checkpoint = if ($Mode -eq "raw") {
        Join-Path $repoRoot "checkpoints\${Dataset}_raw_seed$Seed.pt"
    } else {
        Join-Path $repoRoot "checkpoints\${Dataset}_q_${Mode}_seed$Seed.pt"
    }
    $resume = (Test-Path -LiteralPath $checkpoint) -and (Test-Path -LiteralPath $metrics)
    Write-QueueLog "start dataset=$Dataset mode=$Mode seed=$Seed resume=$resume"
    if ($Mode -eq "raw") {
        & (Join-Path $PSScriptRoot "run_fgvc_raw.ps1") -Dataset $Dataset -Seed $Seed -Steps $Steps -DataRoot $dataRoot -Resume:$resume
    } else {
        & (Join-Path $PSScriptRoot "run_fgvc_q.ps1") -Dataset $Dataset -Seed $Seed -Steps $Steps -DataRoot $dataRoot -QMode $Mode -Resume:$resume
    }
    Write-QueueLog "complete dataset=$Dataset mode=$Mode seed=$Seed"
}

Write-QueueLog "followup queue begin: datasets=eurosat,dtd seeds=$FirstSeed..$LastSeed"
foreach ($dataset in @("eurosat", "dtd")) {
    for ($seed = $FirstSeed; $seed -le $LastSeed; $seed++) {
        foreach ($mode in @("raw", "abs", "signed")) {
            Run-One $dataset $seed $mode
        }
    }
}
Write-QueueLog "followup queue complete"
