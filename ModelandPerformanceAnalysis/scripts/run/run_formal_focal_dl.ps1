param(
    [string]$PythonBin = "C:\Users\Artun\anaconda3\envs\pyspark_env\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonBin)) {
    throw "Python interpreter not found: $PythonBin"
}

$analysisRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repoRoot = Split-Path -Parent $analysisRoot
Set-Location $repoRoot

$datasets = @(
    "hrss_anomalous_optimized",
    "hrss_smote_optimized",
    "hrss_undersample_optimized"
)

$models = @(
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "cnn_lstm_hybrid"
)

foreach ($model in $models) {
    $summaryPath = "ModelandPerformanceAnalysis/results/experiments/results_focal_dl/$model/summary.csv"
    if (Test-Path $summaryPath) {
        Write-Host "Skipping $model because summary already exists at $summaryPath"
        continue
    }

    & $PythonBin "ModelandPerformanceAnalysis/src/main/python/run_experiments.py" `
        --models $model `
        --datasets $datasets `
        --runs 3 `
        --runtime-config "ModelandPerformanceAnalysis/configs/runtime/formal_runtime_config_focal_dl.json" `
        --output-dir "ModelandPerformanceAnalysis/results/experiments/results_focal_dl/$model"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
