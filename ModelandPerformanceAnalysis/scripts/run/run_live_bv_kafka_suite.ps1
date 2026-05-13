$ErrorActionPreference = 'Stop'

function Log([string]$Message) {
    Write-Host "[live-bv] $Message"
}

function Fail([string]$Message) {
    Write-Error "[live-bv] ERROR: $Message"
    exit 1
}

function Get-RepoRoot {
    if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        $scriptDir = $PSScriptRoot
    } elseif (-not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        $scriptDir = Split-Path -Parent $PSCommandPath
    } elseif ($MyInvocation -and $MyInvocation.MyCommand -and -not [string]::IsNullOrWhiteSpace($MyInvocation.MyCommand.Path)) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    } else {
        throw "Could not determine script directory for run_live_bv_kafka_suite.ps1"
    }
    return [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\..\.."))
}

function Get-EnvOrDefault([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

function Resolve-PythonExecutable([string]$Preferred, [string]$Label) {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Preferred)) {
        [void]$candidates.Add($Preferred)
    }

    $condaPrefix = [Environment]::GetEnvironmentVariable('CONDA_PREFIX')
    if (-not [string]::IsNullOrWhiteSpace($condaPrefix)) {
        [void]$candidates.Add((Join-Path $condaPrefix 'python.exe'))
    }

    foreach ($name in @('python', 'python.exe')) {
        try {
            $command = Get-Command $name -ErrorAction SilentlyContinue
            if ($command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
                [void]$candidates.Add($command.Source)
            }
        } catch {
        }
    }

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path $candidate) {
            return (Get-Item $candidate).FullName
        }
        try {
            $command = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($command -and -not [string]::IsNullOrWhiteSpace($command.Source) -and (Test-Path $command.Source)) {
                return (Get-Item $command.Source).FullName
            }
        } catch {
        }
    }

    $candidateList = ($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join '; '
    Fail "Could not resolve $Label Python executable. Checked: $candidateList"
}

function Test-Broker([string]$BootstrapServers) {
    $first = $BootstrapServers.Split(',')[0].Trim()
    if ($first.Contains(':')) {
        $brokerHost, $portText = $first.Split(':', 2)
    } else {
        $brokerHost = $first
        $portText = '9092'
    }
    $port = [int]$portText

    Log "Checking Kafka broker reachability at ${brokerHost}:${port}"
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($brokerHost, $port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(2000, $false)) {
            throw "timeout"
        }
        $client.EndConnect($async) | Out-Null
        Write-Host "Kafka broker reachable at ${brokerHost}:${port}"
    } catch {
        Fail "Kafka broker unreachable at ${brokerHost}:${port}: $($_.Exception.Message)"
    } finally {
        $client.Dispose()
    }
}

function Assert-FreshClasses([string]$RepoRoot) {
    $sources = @(
        Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\KafkaConsumerRF.scala'
        Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\KafkaProducerRF.scala'
        Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\RandomForestModelExport.scala'
        Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\KafkaAnomalyDetection.scala'
    )
    $classes = @(
        Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes\KafkaConsumerRF.class'
        Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes\KafkaConsumerRF$.class'
        Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes\KafkaProducerRF.class'
        Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes\KafkaProducerRF$.class'
        Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes\RandomForestModelExport.class'
        Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes\RandomForestModelExport$.class'
    )

    foreach ($path in $classes) {
        if (-not (Test-Path $path)) {
            return $false
        }
    }

    $latestSource = ($sources | ForEach-Object { (Get-Item $_).LastWriteTimeUtc.Ticks } | Measure-Object -Maximum).Maximum
    $oldestClass = ($classes | ForEach-Object { (Get-Item $_).LastWriteTimeUtc.Ticks } | Measure-Object -Minimum).Minimum
    return ($oldestClass -ge $latestSource)
}

function Compile-ScalaSources([string]$RepoRoot, [string]$JavaHome, [string]$SparkHome) {
    $classesDir = Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11\classes'
    $javaExe = Join-Path $JavaHome 'bin\java.exe'
    $ivyJars = Join-Path $env:USERPROFILE '.ivy2\jars\*'

    $pysparkCandidates = @(
        (Get-EnvOrDefault 'PYSPARK_JARS_DIR' ''),
        'C:\Users\Artun\anaconda3\Lib\site-packages\pyspark\jars',
        'C:\Users\artun\anaconda3\Lib\site-packages\pyspark\jars'
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    $pysparkJarDir = $pysparkCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $pysparkJarDir) {
        Fail 'Could not locate a PySpark jars directory for Scala compilation.'
    }

    $scalaCompiler = Join-Path $pysparkJarDir 'scala-compiler-2.11.8.jar'
    $scalaLibrary  = Join-Path $pysparkJarDir 'scala-library-2.11.8.jar'
    $scalaReflect  = Join-Path $pysparkJarDir 'scala-reflect-2.11.8.jar'
    foreach ($jar in @($scalaCompiler, $scalaLibrary, $scalaReflect)) {
        if (-not (Test-Path $jar)) {
            Fail "Missing Scala compiler dependency: $jar"
        }
    }
    if (-not (Test-Path $javaExe)) {
        Fail "java.exe not found under JAVA_HOME: $javaExe"
    }

    New-Item -ItemType Directory -Force -Path $classesDir | Out-Null

    $compileClasspath = @(
        $classesDir,
        (Join-Path $SparkHome 'jars\*'),
        (Join-Path $pysparkJarDir '*'),
        $ivyJars
    ) -join ';'

    $launcherClasspath = @($scalaCompiler, $scalaLibrary, $scalaReflect) -join ';'

    $sources = @(
        (Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\KafkaConsumerRF.scala'),
        (Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\KafkaProducerRF.scala'),
        (Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\RandomForestModelExport.scala'),
        (Join-Path $RepoRoot 'SparkKafkaStreaming\src\main\scala\spark\KafkaAnomalyDetection.scala')
    )

    Log 'Refreshing Scala classes with local scala.tools.nsc compiler'
    & $javaExe `
        -cp $launcherClasspath `
        scala.tools.nsc.Main `
        -classpath $compileClasspath `
        -d $classesDir `
        $sources

    if ($LASTEXITCODE -ne 0) {
        Fail "Local Scala compilation failed with exit code $LASTEXITCODE."
    }
}

function Ensure-RuntimeArtifact([string]$RepoRoot, [string]$JavaHome) {
    $targetDir = Join-Path $RepoRoot 'SparkKafkaStreaming\target\scala-2.11'
    $classesDir = Join-Path $targetDir 'classes'
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $jarPath = Join-Path $targetDir ("effitrack-live-thin-$timestamp-$PID.jar")
    $jarExe = Join-Path $JavaHome 'bin\jar.exe'

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $sbt = Get-Command sbt -ErrorAction SilentlyContinue
    if ($sbt) {
        Log 'Refreshing Scala classes with sbt compile'
        Push-Location (Join-Path $RepoRoot 'SparkKafkaStreaming')
        try {
            & $sbt.Source compile
        } finally {
            Pop-Location
        }
    } elseif ((Get-EnvOrDefault 'ALLOW_STALE_CLASSES' '0') -ne '1') {
        if (-not (Assert-FreshClasses $RepoRoot)) {
            Compile-ScalaSources $RepoRoot $JavaHome $SparkHome
            if (-not (Assert-FreshClasses $RepoRoot)) {
                Fail "Scala class files are older than the edited source files. Automatic local compilation did not refresh them."
            }
        }
    } else {
        Log 'ALLOW_STALE_CLASSES=1 set, using existing compiled classes'
    }

    if (-not (Test-Path $classesDir)) {
        Fail "Missing compiled classes directory: $classesDir"
    }
    if (-not (Test-Path $jarExe)) {
        Fail "jar.exe not found under JAVA_HOME: $jarExe"
    }

    Log "Packaging runtime jar: $jarPath"
    & $jarExe cf $jarPath -C $classesDir .
    return $jarPath
}

function Start-Consumer([string]$RepoRoot, [string]$SparkSubmitCmd, [string]$RuntimeJar, [string]$StdoutLogFile, [string]$StderrLogFile, [string]$SparkMaster, [string]$SparkPackages) {
    $argumentList = @(
        '--master', $SparkMaster,
        '--packages', $SparkPackages,
        '--class', 'KafkaConsumerRF',
        $RuntimeJar
    )

    Add-Content -Path $StdoutLogFile -Value ("COMMAND: " + $SparkSubmitCmd + " " + ($argumentList -join ' '))

    return Start-Process -FilePath $SparkSubmitCmd `
        -WorkingDirectory $RepoRoot `
        -ArgumentList $argumentList `
        -RedirectStandardOutput $StdoutLogFile `
        -RedirectStandardError $StderrLogFile `
        -PassThru `
        -WindowStyle Hidden
}

function Start-Producer([string]$RepoRoot, [string]$SparkSubmitCmd, [string]$RuntimeJar, [string]$StdoutLogFile, [string]$StderrLogFile, [string]$SparkPackages) {
    $argumentList = @(
        '--master', 'local[1]',
        '--packages', $SparkPackages,
        '--class', 'KafkaProducerRF',
        $RuntimeJar
    )

    Add-Content -Path $StdoutLogFile -Value ("COMMAND: " + $SparkSubmitCmd + " " + ($argumentList -join ' '))

    return Start-Process -FilePath $SparkSubmitCmd `
        -WorkingDirectory $RepoRoot `
        -ArgumentList $argumentList `
        -RedirectStandardOutput $StdoutLogFile `
        -RedirectStandardError $StderrLogFile `
        -PassThru `
        -WindowStyle Hidden
}

function Read-LogText([string]$Path) {
    if (-not (Test-Path $Path)) {
        return ''
    }
    return Get-Content $Path -Raw -ErrorAction SilentlyContinue
}

function Get-ProcessExitCodeOrNull($Process) {
    if ($null -eq $Process) {
        return $null
    }
    try {
        $Process.Refresh()
    } catch {
    }
    try {
        if ($Process.HasExited) {
            return $Process.ExitCode
        }
    } catch {
    }
    return $null
}

function Test-ProducerCompletion([string]$StdoutLogFile) {
    $stdout = Read-LogText $StdoutLogFile
    return ($stdout -match 'KafkaProducerRF replay completed\.')
}

function Wait-ForConsumerStart($Process, [string]$StdoutLogFile, [string]$StderrLogFile, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $stdout = Read-LogText $StdoutLogFile
        $stderr = Read-LogText $StderrLogFile

        if ($stdout -match 'KafkaConsumerRF started with inputTopic=') {
            return $true
        }

        if (($stdout + "`n" + $stderr) -match 'Exception|ERROR|error:') {
            return $false
        }

        if ($Process -and $Process.HasExited) {
            Add-Content -Path $StdoutLogFile -Value ("PROCESS_EXIT_CODE: " + $Process.ExitCode)
            if (-not [string]::IsNullOrWhiteSpace($stderr)) {
                Add-Content -Path $StdoutLogFile -Value "STDERR_BEGIN"
                Add-Content -Path $StdoutLogFile -Value $stderr
                Add-Content -Path $StdoutLogFile -Value "STDERR_END"
            }
            if ($Process.ExitCode -eq 0 -and $stdout -match 'KafkaConsumerRF started with inputTopic=') {
                return $true
            } else {
                return $false
            }
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Stop-ProcessTree($Process) {
    if ($null -eq $Process) {
        return
    }
    try {
        $Process.Refresh()
    } catch {
    }
    try {
        if ($Process.HasExited) {
            return
        }
    } catch {
    }
    try {
        cmd.exe /c "taskkill /PID $($Process.Id) /T /F >NUL 2>NUL" | Out-Null
    } catch {
        try {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        } catch {
        }
    }
}

$RepoRoot = Get-RepoRoot
Set-Location $RepoRoot

$JavaHome = Get-EnvOrDefault 'JAVA_HOME' 'C:\Program Files\Java\jdk1.8.0_202'
$SparkHome = Get-EnvOrDefault 'SPARK_HOME' 'C:\spark'
$PythonExe = Resolve-PythonExecutable (Get-EnvOrDefault 'PYSPARK_PYTHON' 'python') 'worker'
$DriverPythonExe = Resolve-PythonExecutable (Get-EnvOrDefault 'PYSPARK_DRIVER_PYTHON' $PythonExe) 'driver'
$SparkSubmitCmd = Join-Path $SparkHome 'bin\spark-submit.cmd'

if (-not (Test-Path $SparkSubmitCmd)) {
    Fail "spark-submit.cmd not found: $SparkSubmitCmd"
}

[Environment]::SetEnvironmentVariable('JAVA_HOME', $JavaHome, 'Process')
[Environment]::SetEnvironmentVariable('SPARK_HOME', $SparkHome, 'Process')
[Environment]::SetEnvironmentVariable('PYSPARK_PYTHON', $PythonExe, 'Process')
[Environment]::SetEnvironmentVariable('PYSPARK_DRIVER_PYTHON', $DriverPythonExe, 'Process')
[Environment]::SetEnvironmentVariable('KAFKA_BOOTSTRAP_SERVERS', (Get-EnvOrDefault 'KAFKA_BOOTSTRAP_SERVERS' 'localhost:9092'), 'Process')
[Environment]::SetEnvironmentVariable('KAFKA_INPUT_TOPIC', (Get-EnvOrDefault 'KAFKA_INPUT_TOPIC' 'model-input'), 'Process')
[Environment]::SetEnvironmentVariable('ANOMALY_TOPIC', (Get-EnvOrDefault 'ANOMALY_TOPIC' 'anomalies3'), 'Process')
[Environment]::SetEnvironmentVariable('NORMAL_TOPIC', (Get-EnvOrDefault 'NORMAL_TOPIC' 'normal_data'), 'Process')
[Environment]::SetEnvironmentVariable('UNCERTAIN_TOPIC', (Get-EnvOrDefault 'UNCERTAIN_TOPIC' 'uncertain_data'), 'Process')
[Environment]::SetEnvironmentVariable('ANOMALY_THRESHOLD', (Get-EnvOrDefault 'ANOMALY_THRESHOLD' '0.8'), 'Process')
[Environment]::SetEnvironmentVariable('NORMAL_THRESHOLD', (Get-EnvOrDefault 'NORMAL_THRESHOLD' '0.2'), 'Process')
[Environment]::SetEnvironmentVariable('STREAM_BATCH_INTERVAL_SEC', (Get-EnvOrDefault 'STREAM_BATCH_INTERVAL_SEC' '10'), 'Process')
[Environment]::SetEnvironmentVariable('REPLAY_FLOW_RATE_MS', (Get-EnvOrDefault 'REPLAY_FLOW_RATE_MS' '10'), 'Process')
[Environment]::SetEnvironmentVariable('MAX_MESSAGES', (Get-EnvOrDefault 'MAX_MESSAGES' '0'), 'Process')
[Environment]::SetEnvironmentVariable('POST_RUN_WAIT_SEC', (Get-EnvOrDefault 'POST_RUN_WAIT_SEC' '5'), 'Process')
[Environment]::SetEnvironmentVariable('SPARK_MASTER', (Get-EnvOrDefault 'SPARK_MASTER' 'local[*]'), 'Process')
[Environment]::SetEnvironmentVariable('CONSUMER_START_TIMEOUT_SEC', (Get-EnvOrDefault 'CONSUMER_START_TIMEOUT_SEC' '180'), 'Process')
[Environment]::SetEnvironmentVariable('RF_MODEL_PATH', (Get-EnvOrDefault 'RF_MODEL_PATH' 'SparkKafkaStreaming\model_artifacts\random_forest_smote_standard_pipeline'), 'Process')
[Environment]::SetEnvironmentVariable('RF_MODEL_METADATA_PATH', (Get-EnvOrDefault 'RF_MODEL_METADATA_PATH' 'SparkKafkaStreaming\model_artifacts\random_forest_smote_standard_pipeline_metadata.json'), 'Process')
[Environment]::SetEnvironmentVariable('RF_TRAINING_DATA_PATH', (Get-EnvOrDefault 'RF_TRAINING_DATA_PATH' 'Data\HRSS_SMOTE_standard.csv'), 'Process')
[Environment]::SetEnvironmentVariable('STREAM_SOURCE_NORMAL_CSV_PATH', (Get-EnvOrDefault 'STREAM_SOURCE_NORMAL_CSV_PATH' 'Data\HRSS_normal_standard.csv'), 'Process')
[Environment]::SetEnvironmentVariable('STREAM_SOURCE_ANOMALOUS_CSV_PATH', (Get-EnvOrDefault 'STREAM_SOURCE_ANOMALOUS_CSV_PATH' 'Data\HRSS_anomalous_standard.csv'), 'Process')
[Environment]::SetEnvironmentVariable('KAFKA_AUTO_OFFSET_RESET', (Get-EnvOrDefault 'KAFKA_AUTO_OFFSET_RESET' 'latest'), 'Process')

$SparkPackages = Get-EnvOrDefault 'SPARK_PACKAGES' 'org.apache.spark:spark-streaming-kafka-0-10_2.11:2.3.1'
$BootstrapServers = [Environment]::GetEnvironmentVariable('KAFKA_BOOTSTRAP_SERVERS')
$SparkMaster = [Environment]::GetEnvironmentVariable('SPARK_MASTER')
$TimeoutSeconds = [int][Environment]::GetEnvironmentVariable('CONSUMER_START_TIMEOUT_SEC')
$PostRunWaitSeconds = [int][Environment]::GetEnvironmentVariable('POST_RUN_WAIT_SEC')
$RFModelPath = [Environment]::GetEnvironmentVariable('RF_MODEL_PATH')

Test-Broker $BootstrapServers

$RuntimeJar = Ensure-RuntimeArtifact $RepoRoot $JavaHome
[Environment]::SetEnvironmentVariable('EFFITRACK_RUNTIME_JAR', $RuntimeJar, 'Process')

if ((-not (Test-Path $RFModelPath)) -or ((Get-EnvOrDefault 'RF_FORCE_REEXPORT' '0') -eq '1')) {
    Log 'Exporting Random Forest pipeline artifact'
    & $SparkSubmitCmd --master 'local[*]' --class RandomForestModelExport $RuntimeJar
} else {
    Log "Using existing Random Forest pipeline artifact: $RFModelPath"
}

$LogDir = Join-Path $RepoRoot 'SparkKafkaStreaming\results\live_metrics'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"

$ConsumerLog    = Join-Path $LogDir "live_consumer_stdout_$RunId.log"
$ConsumerErrLog = Join-Path $LogDir "live_consumer_stderr_$RunId.log"
$ProducerLog    = Join-Path $LogDir "live_producer_stdout_$RunId.log"
$ProducerErrLog = Join-Path $LogDir "live_producer_stderr_$RunId.log"
$MetricsLog     = Join-Path $LogDir "live_kafka_metrics_$RunId.jsonl"

[Environment]::SetEnvironmentVariable('LIVE_METRICS_PATH', $MetricsLog, 'Process')
if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable('KAFKA_CONSUMER_GROUP'))) {
    [Environment]::SetEnvironmentVariable('KAFKA_CONSUMER_GROUP', "rf-group-$RunId", 'Process')
}

New-Item -ItemType File -Path $ConsumerLog -Force | Out-Null
New-Item -ItemType File -Path $ConsumerErrLog -Force | Out-Null
New-Item -ItemType File -Path $ProducerLog -Force | Out-Null
New-Item -ItemType File -Path $ProducerErrLog -Force | Out-Null
New-Item -ItemType File -Path $MetricsLog -Force | Out-Null

Write-Host "[live-bv] Log run id: $RunId"
Write-Host "[live-bv] Consumer stdout: $ConsumerLog"
Write-Host "[live-bv] Consumer stderr: $ConsumerErrLog"
Write-Host "[live-bv] Producer stdout: $ProducerLog"
Write-Host "[live-bv] Producer stderr: $ProducerErrLog"
Write-Host "[live-bv] Metrics log: $MetricsLog"

$consumerProcess = $null
$producerProcess = $null
try {
    Log 'Starting live consumer'
    $consumerProcess = Start-Consumer $RepoRoot $SparkSubmitCmd $RuntimeJar $ConsumerLog $ConsumerErrLog $SparkMaster $SparkPackages

    if (-not (Wait-ForConsumerStart $consumerProcess $ConsumerLog $ConsumerErrLog $TimeoutSeconds)) {
        if (Test-Path $ConsumerLog) {
            Get-Content $ConsumerLog -TotalCount 200
        }
        if (Test-Path $ConsumerErrLog) {
            Get-Content $ConsumerErrLog -TotalCount 200
        }
        Fail 'KafkaConsumerRF did not start cleanly.'
    }

    Log 'Starting raw-data producer'
    $producerProcess = Start-Producer $RepoRoot $SparkSubmitCmd $RuntimeJar $ProducerLog $ProducerErrLog $SparkPackages
    $producerProcess.WaitForExit()
    $producerExitCode = Get-ProcessExitCodeOrNull $producerProcess
    $producerExitCodeText = if ($null -eq $producerExitCode) { 'unknown' } else { [string]$producerExitCode }
    $producerCompleted = Test-ProducerCompletion $ProducerLog
    Add-Content -Path $ProducerLog -Value ("PROCESS_EXIT_CODE: " + $producerExitCodeText)

    if (-not $producerCompleted) {
        if (Test-Path $ProducerLog) {
            Get-Content $ProducerLog -TotalCount 200
        }
        if (Test-Path $ProducerErrLog) {
            Get-Content $ProducerErrLog -TotalCount 200
        }
        Fail "KafkaProducerRF did not reach its completion marker. ExitCode=$producerExitCodeText."
    }

    if ($null -ne $producerExitCode -and $producerExitCode -ne 0) {
        Log "Producer completion marker found; ignoring non-zero exit code $producerExitCodeText from spark-submit.cmd wrapper"
    }

    Log "Waiting ${PostRunWaitSeconds}s for consumer to flush live metrics"
    Start-Sleep -Seconds $PostRunWaitSeconds
} finally {
    Log 'Stopping live consumer'
    Stop-ProcessTree $producerProcess
    Stop-ProcessTree $consumerProcess
}

if (Test-Path $MetricsLog) {
    Log "Live metrics log created: $MetricsLog"
} else {
    Log 'Live metrics log was not created; suite will report this as missing'
}

Log 'Running BV suite summary'
& $PythonExe (Join-Path $RepoRoot 'ModelandPerformanceAnalysis\scripts\run\run_bv_full_suite.py')
