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

$outputRoot = "ModelandPerformanceAnalysis/results/experiments/results_generalization_suite"
$summaryPath = "ModelandPerformanceAnalysis/results/summaries/generalization_suite_summary_all.csv"

Remove-Item $outputRoot -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $summaryPath -Force -ErrorAction SilentlyContinue

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/run_generalization_suite.py" `
    --output-dir $outputRoot `
    --summary-output $summaryPath `
    --models all `
    --runtime-config "ModelandPerformanceAnalysis/configs/runtime/formal_runtime_config_v2.json"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
