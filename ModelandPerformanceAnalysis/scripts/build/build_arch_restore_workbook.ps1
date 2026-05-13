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

$outputRoot = "ModelandPerformanceAnalysis/results/experiments/results_arch_restore_rerun"
$summaryPath = "ModelandPerformanceAnalysis/results/summaries/formal_arch_restore_summary_all.csv"
$workbookPath = "ModelandPerformanceAnalysis/results/workbooks/ModelResultsArchitecturesRestored.xlsx"

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/merge_experiment_results.py" `
    --summary-csv "ModelandPerformanceAnalysis/results/experiments/results_formal_new/classical/summary.csv" `
    --json-root "$outputRoot/cnn" `
    --json-root "$outputRoot/rnn" `
    --json-root "$outputRoot/lstm" `
    --json-root "$outputRoot/gru" `
    --json-root "$outputRoot/autoencoder" `
    --json-root "$outputRoot/vanilla_transformer" `
    --json-root "$outputRoot/encoder_decoder_transformer" `
    --json-root "$outputRoot/temporal_fusion_transformer" `
    --json-root "$outputRoot/cnn_lstm_hybrid" `
    --output $summaryPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/export_results_workbook.py" `
    --summary $summaryPath `
    --output $workbookPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
