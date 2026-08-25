param(
    [int]$FirstPairedSeed = 2,
    [int]$LastPairedSeed = 5,
    [int]$Steps = 3000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path (Split-Path -Parent $repoRoot) "data"
$log = Join-Path $repoRoot "results\multiseed_queue.log"

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

function Run-Raw([int]$Seed) {
    $metrics = Join-Path $repoRoot "results\fgvc_seed$Seed\raw_$Steps.csv"
    $checkpoint = Join-Path $repoRoot "checkpoints\fgvc_raw_seed$Seed.pt"
    if ((Last-Step $metrics) -ge $Steps) {
        Write-QueueLog "skip completed raw seed=$Seed"
        return
    }
    $resume = (Test-Path -LiteralPath $checkpoint) -and (Test-Path -LiteralPath $metrics)
    Write-QueueLog "start raw seed=$Seed resume=$resume"
    & (Join-Path $PSScriptRoot "run_fgvc_raw.ps1") -Seed $Seed -Steps $Steps -DataRoot $dataRoot -Resume:$resume
    Write-QueueLog "complete raw seed=$Seed"
}

function Run-Q([int]$Seed, [string]$Mode) {
    $metrics = Join-Path $repoRoot "results\fgvc_seed$Seed\q_${Mode}_$Steps.csv"
    $checkpoint = Join-Path $repoRoot "checkpoints\fgvc_q_${Mode}_seed$Seed.pt"
    if ((Last-Step $metrics) -ge $Steps) {
        Write-QueueLog "skip completed q_$Mode seed=$Seed"
        return
    }
    $resume = (Test-Path -LiteralPath $checkpoint) -and (Test-Path -LiteralPath $metrics)
    Write-QueueLog "start q_$Mode seed=$Seed resume=$resume"
    & (Join-Path $PSScriptRoot "run_fgvc_q.ps1") -Seed $Seed -Steps $Steps -DataRoot $dataRoot -QMode $Mode -Resume:$resume
    Write-QueueLog "complete q_$Mode seed=$Seed"
}

New-Item -ItemType Directory -Force (Join-Path $repoRoot "results") | Out-Null
Write-QueueLog "queue begin: signed seed=1; paired seeds=$FirstPairedSeed..$LastPairedSeed"
Run-Q 1 "signed"
for ($seed = $FirstPairedSeed; $seed -le $LastPairedSeed; $seed++) {
    Run-Raw $seed
    Run-Q $seed "abs"
    Run-Q $seed "signed"
}
Write-QueueLog "queue complete"
