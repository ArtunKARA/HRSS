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

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/run_experiments.py" `
    --models cnn rnn lstm gru autoencoder vanilla_transformer encoder_decoder_transformer temporal_fusion_transformer cnn_lstm_hybrid `
    --datasets hrss_anomalous_optimized hrss_smote_optimized `
    --runs 1 `
    --runtime-config "ModelandPerformanceAnalysis/configs/runtime/smoke_runtime_config_data_driven.json" `
    --output-dir "ModelandPerformanceAnalysis/results/experiments/smoke_data_driven_v2"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
