param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [string]$Destination = (Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads\prooftag-benchmarks"),
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$remoteCommand = "cd $RemoteRepository && make benchmark"
$remoteOutput = & ssh $Server $remoteCommand | Tee-Object -Variable capturedOutput
if ($LASTEXITCODE -ne 0) {
    throw "Le benchmark distant a échoué avec le code $LASTEXITCODE."
}

$archiveLine = $capturedOutput |
    Where-Object { $_ -like "BENCHMARK_ARCHIVE=*" } |
    Select-Object -Last 1
if (-not $archiveLine) {
    throw "Le benchmark n'a pas retourné le chemin de son archive."
}

$remoteArchive = $archiveLine.Substring("BENCHMARK_ARCHIVE=".Length)
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
Write-Host "Rapport copié : $report"

if (-not $NoOpen) {
    Start-Process $report
}
