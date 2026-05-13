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

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/prepare_selected_datasets.py" `
    --real-source "real_imbalanced" `
    --output-dir "Data/benchmark_suite"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
