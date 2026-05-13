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

$workbookPath = "ModelandPerformanceAnalysis/results/workbooks/CoreDatasetFamilySummary.xlsx"

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/export_dataset_overview_workbook.py" `
    --output $workbookPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
