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

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/merge_experiment_results.py" `
    --summary-csv "ModelandPerformanceAnalysis/results/experiments/results_formal_new/classical/summary.csv" `
    --json-root "ModelandPerformanceAnalysis/results/experiments/results_formal_models" `
    --json-root "ModelandPerformanceAnalysis/results/experiments/results_formal_standardized" `
    --json-root "ModelandPerformanceAnalysis/results/experiments/results_focal_dl" `
    --output "ModelandPerformanceAnalysis/results/summaries/formal_focal_dl_summary_all.csv"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/export_results_workbook.py" `
    --summary "ModelandPerformanceAnalysis/results/summaries/formal_focal_dl_summary_all.csv" `
    --output "ModelandPerformanceAnalysis/results/workbooks/ModelResultsFocal.xlsx"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
