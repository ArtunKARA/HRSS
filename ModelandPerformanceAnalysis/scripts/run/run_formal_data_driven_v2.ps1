param(
    [string]$PythonBin = "C:\Users\Artun\anaconda3\envs\pyspark_env\python.exe",
    [switch]$Resume
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
    "hrss_undersample_optimized",
    "hrss_smote_optimized",
    "hrss_anomalous_standard",
    "hrss_undersample_standard",
    "hrss_smote_standard"
)

$models = @(
    "logistic_regression",
    "random_forest",
    "svm",
    "decision_tree",
    "knn",
    "naive_bayes",
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

$outputRoot = "ModelandPerformanceAnalysis/results/experiments/results_data_driven_v2_formal"

if (-not $Resume) {
    Remove-Item $outputRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not $Resume) {
    & $PythonBin "ModelandPerformanceAnalysis/src/main/python/prepare_core_dataset_families.py"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$expectedDatasets = $datasets.Count

foreach ($model in $models) {
    $modelDir = Join-Path (Join-Path $repoRoot $outputRoot) $model
    $summaryPath = Join-Path $modelDir "summary.csv"

    if ($Resume -and (Test-Path $summaryPath)) {
        $summaryRows = @(Import-Csv -Path $summaryPath)
        if ($summaryRows.Count -ge $expectedDatasets) {
            Write-Host "Skipping $model (summary.csv has $($summaryRows.Count) rows, expected $expectedDatasets)."
            continue
        }
    }

    Write-Host "Running $model ..."
    & $PythonBin "ModelandPerformanceAnalysis/src/main/python/run_experiments.py" `
        --models $model `
        --datasets $datasets `
        --runs 3 `
        --runtime-config "ModelandPerformanceAnalysis/configs/runtime/formal_runtime_config_v2.json" `
        --output-dir "$outputRoot/$model"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
