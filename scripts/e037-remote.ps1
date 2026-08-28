param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18888,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$Notebook = "32_e037_prospective_global_trust_holdout.ipynb"
$pidFile = Join-Path $env:TEMP "prooftag-qr-e037-tunnel.pid"
$logFile = Join-Path $env:TEMP "prooftag-qr-e037-ssh.log"

function Stop-E037Tunnel {
    if (-not (Test-Path $pidFile)) { return }
    $tunnelPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $process = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -eq "ssh") {
        Stop-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

function Assert-LocalPortAvailable {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $LocalPort
    )
    try { $listener.Start() }
    catch { throw "Le port local $LocalPort est deja occupe. Utilise -LocalPort 18889." }
    finally { $listener.Stop() }
}

function Invoke-RemoteBash {
    param([Parameter(Mandatory = $true)][string]$Script)
    $normalized = $Script.Replace("`r`n", "`n").Replace("`r", "`n")
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $base64 = [Convert]::ToBase64String($utf8.GetBytes($normalized))
    $remoteCommand = "printf '%s' '$base64' | base64 --decode | bash"
    $output = & ssh $Server $remoteCommand
    if ($LASTEXITCODE -ne 0) { throw "Execution Bash distante E037 en echec." }
    return $output
}

if ($Stop) {
    Stop-E037Tunnel
    Write-Host "Tunnel E037 arrete."
    exit 0
}

Stop-E037Tunnel
Assert-LocalPortAvailable

$remoteScript = @"
set -Eeuo pipefail
cd $RemoteRepository
NS="`${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="`${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
NOTEBOOK="$Notebook"
RESULTS="/data/e037-prospective-mini-holdout-v1"
MODE="`$(kubectl get deployment "`$DEP" -n "`$NS" -o jsonpath='{.spec.template.metadata.annotations.prooftag\.io/notebook-mode}')"
if [ "`$MODE" != "advisor-cpu" ]; then
  echo "E037 notebook mode invalide: `$MODE" >&2
  exit 1
fi
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "/workspace/notebooks/`$NOTEBOOK"
if ! kubectl exec -n "`$NS" deployment/prooftag-qr -c api -- test -f "`$RESULTS/verdict.json"; then
  echo "E037 n'est pas encore termine. Lancer: bash scripts/run-e037-holdout.sh" >&2
  exit 1
fi
ENC="`$(kubectl get secret prooftag-qr-notebook -n "`$NS" -o jsonpath='{.data.token}')"
TOKEN="`$(printf '%s' "`$ENC" | base64 --decode)"
IP="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.clusterIP}')"
PORT="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.ports[0].port}')"
printf 'JUPYTER_TOKEN=%s\n' "`$TOKEN"
printf 'JUPYTER_TARGET=%s:%s\n' "`$IP" "`$PORT"
"@

try { $remoteOutput = Invoke-RemoteBash -Script $remoteScript }
catch { throw ("Notebook E037 distant indisponible. " + $_.Exception.Message) }

$tokenLine = $remoteOutput | Where-Object { $_ -like "JUPYTER_TOKEN=*" } | Select-Object -Last 1
$targetLine = $remoteOutput | Where-Object { $_ -like "JUPYTER_TARGET=*" } | Select-Object -Last 1
if (-not $tokenLine -or -not $targetLine) { throw "Token/cible Jupyter E037 absents." }
$token = $tokenLine.Substring("JUPYTER_TOKEN=".Length)
$target = $targetLine.Substring("JUPYTER_TARGET=".Length)
if ($target -notmatch '^\d{1,3}(\.\d{1,3}){3}:\d+$') { throw "Cible Jupyter invalide : $target" }

Remove-Item -LiteralPath $logFile -Force -ErrorAction SilentlyContinue
$arguments = @(
    "-N", "-v", "-E", $logFile,
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-L", "${LocalPort}:${target}",
    $Server
)
Write-Host "Une fenetre SSH va s'ouvrir : saisir le mot de passe de $Server."
$tunnel = Start-Process -FilePath "ssh" -ArgumentList $arguments -WindowStyle Normal -PassThru
Set-Content -LiteralPath $pidFile -Value $tunnel.Id

$url = "http://127.0.0.1:${LocalPort}/lab/tree/notebooks/${Notebook}?token=$token"
$headers = @{ Authorization = "token $token" }
for ($attempt = 0; $attempt -lt 240; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:${LocalPort}/api" -Headers $headers -TimeoutSec 1 | Out-Null
        Start-Process $url
        Write-Host "E037 ouvert : $url"
        Write-Host "Le notebook affiche le contact sheet global et les 10 comparaisons parent/final."
        Write-Host "Pour fermer le tunnel : .\scripts\e037-remote.ps1 -Stop"
        exit 0
    }
    catch {
        if ($tunnel.HasExited) { throw "Tunnel SSH E037 arrete. Log: $logFile" }
    }
}
throw "Jupyter E037 n'a pas repondu sur le port $LocalPort."
