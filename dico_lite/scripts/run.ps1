param (
    [string]$RunTrain = "false"
)

# Navigate to the script's parent directory, then up one level to the dico_lite root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

Write-Host "Running DiCo-lite Comparative Experiment Suite" -ForegroundColor Green

Write-Host "`n[1/2] Running Uniform Baseline..." -ForegroundColor Cyan
python -m src.main --config configs/config.json --method uniform --run_train $RunTrain

Write-Host "`n[2/2] Running Module-DiCo-lite Baseline..." -ForegroundColor Cyan
python -m src.main --config configs/config.json --method dico --run_train $RunTrain

Write-Host "`nComparative Suite Complete!" -ForegroundColor Green
