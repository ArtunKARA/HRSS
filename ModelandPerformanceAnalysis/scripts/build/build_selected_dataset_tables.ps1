param(
    [string]$PythonBin = "C:\Users\Artun\anaconda3\envs\pyspark_env\python.exe",
    [ValidateSet("tr", "en")]
    [string]$Locale = "tr"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonBin)) {
    throw "Python interpreter not found: $PythonBin"
}

$analysisRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repoRoot = Split-Path -Parent $analysisRoot
Set-Location $repoRoot

$tablesDir = "ModelandPerformanceAnalysis/results/tables"
$csvPath = "$tablesDir/core_dataset_family_overview_$Locale.csv"
$markdownPath = "$tablesDir/core_dataset_family_overview_$Locale.md"
$workbookPath = "ModelandPerformanceAnalysis/results/workbooks/CoreDatasetFamilySummary_$($Locale.ToUpper()).xlsx"

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/export_dataset_overview_table.py" `
    --locale $Locale `
    --csv-output $csvPath `
    --markdown-output $markdownPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonBin "ModelandPerformanceAnalysis/src/main/python/export_dataset_overview_workbook.py" `
    --locale $Locale `
    --output $workbookPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
