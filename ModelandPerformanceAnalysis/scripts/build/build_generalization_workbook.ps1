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

$summaryPath = "ModelandPerformanceAnalysis/results/summaries/generalization_suite_summary_all.csv"
$workbookPath = "ModelandPerformanceAnalysis/results/workbooks/ModelResultsGeneralization.xlsx"

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/export_results_workbook.py" `
    --summary $summaryPath `
    --output $workbookPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
