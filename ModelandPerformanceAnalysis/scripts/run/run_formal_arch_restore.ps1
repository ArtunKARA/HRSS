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
    "autoencoder",
    "vanilla_transformer",
    "encoder_decoder_transformer",
    "temporal_fusion_transformer",
    "cnn_lstm_hybrid"
)

$outputRoot = "ModelandPerformanceAnalysis/results/experiments/results_arch_restore_rerun"
Remove-Item $outputRoot -Recurse -Force -ErrorAction SilentlyContinue

foreach ($model in $models) {
    & $PythonBin "ModelandPerformanceAnalysis/src/main/python/run_experiments.py" `
        --models $model `
        --datasets $datasets `
        --runs 3 `
        --runtime-config "ModelandPerformanceAnalysis/configs/runtime/formal_runtime_config.json" `
        --output-dir "$outputRoot/$model"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
