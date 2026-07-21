param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [string]$Destination = (Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads\prooftag-benchmarks"),
    [switch]$E004,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$makeTarget = if ($E004) { "benchmark-e004" } else { "benchmark" }
$remoteCommand = "cd $RemoteRepository && make $makeTarget"
$remoteOutput = & ssh $Server $remoteCommand | Tee-Object -Variable capturedOutput
if ($LASTEXITCODE -ne 0) {
    throw "Le benchmark distant a échoué avec le code $LASTEXITCODE."
}

$archivePrefixes = if ($E004) {
    @("E004_BASELINE_ARCHIVE=", "E004_GUIDED_ARCHIVE=")
} else {
    @("BENCHMARK_ARCHIVE=")
}

$reports = @()
foreach ($archivePrefix in $archivePrefixes) {
    $archiveLine = $capturedOutput |
        Where-Object { $_ -like "$archivePrefix*" } |
        Select-Object -Last 1
    if (-not $archiveLine) {
        throw "Le benchmark n'a pas retourné la valeur $archivePrefix."
    }

    $remoteArchive = $archiveLine.Substring($archivePrefix.Length)
    $archiveName = Split-Path -Leaf $remoteArchive
    $localArchive = Join-Path $Destination $archiveName

    & scp "${Server}:$remoteArchive" $localArchive
    if ($LASTEXITCODE -ne 0) {
        throw "Le transfert SCP a échoué avec le code $LASTEXITCODE."
    }

    & tar -xzf $localArchive -C $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "L'extraction de l'archive a échoué avec le code $LASTEXITCODE."
    }

    $runName = $archiveName -replace '\.tar\.gz$', ''
    $report = Join-Path (Join-Path $Destination $runName) "report.html"
    $reports += $report
    Write-Host "Rapport copié : $report"
}

if (-not $NoOpen) {
    Start-Process $reports[-1]
}
