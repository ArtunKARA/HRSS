param(
    [string]$PythonBin = "C:\Users\Artun\anaconda3\envs\pyspark_env\python.exe",
    [int]$Runs = 10,
    [int]$CvFolds = 10,
    [string[]]$Datasets = @(
        "real_imbalanced",
        "real_downsample_balanced",
        "real_smote_balanced"
    ),
    [string[]]$Models = @(
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
    ),
    [string]$RuntimeConfig = "ModelandPerformanceAnalysis/configs/runtime/selected_real_dataset_runtime_config.json",
    [string]$OutputRoot = "ModelandPerformanceAnalysis/results/experiments/results_selected_real_dataset_suite",
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonBin)) {
    throw "Python interpreter not found: $PythonBin"
}

$analysisRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repoRoot = Split-Path -Parent $analysisRoot
Set-Location $repoRoot

if (-not $SkipPrepare) {
    & $PythonBin "ModelandPerformanceAnalysis/src/main/python/prepare_selected_datasets.py" `
        --real-source "real_imbalanced" `
        --output-dir "Data/benchmark_suite"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
Remove-Item $outputRoot -Recurse -Force -ErrorAction SilentlyContinue

foreach ($model in $Models) {
    & $PythonBin "ModelandPerformanceAnalysis/src/main/python/run_experiments.py" `
        --models $model `
        --datasets $Datasets `
        --runs $Runs `
        --cv-folds $CvFolds `
        --runtime-config $RuntimeConfig `
        --output-dir "$outputRoot/$model"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
