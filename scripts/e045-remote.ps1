param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18945,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$Notebook = "47_e045_foundation_and_resilience.ipynb"
$pidFile = Join-Path $env:TEMP "prooftag-qr-e045-tunnel.pid"
$logFile = Join-Path $env:TEMP "prooftag-qr-e045-ssh.log"

function Stop-Tunnel {
    if (-not (Test-Path $pidFile)) {
        return
    }
    $tunnelPid = [int](Get-Content $pidFile -Raw)
    $process = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -eq "ssh") {
        Stop-Process -Id $tunnelPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

function Assert-PortAvailable {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $LocalPort
    )
    try {
        $listener.Start()
    }
    catch {
        throw "Port $LocalPort occupe. Utilise -LocalPort $($LocalPort + 1)."
    }
    finally {
        $listener.Stop()
    }
}

if ($Stop) {
    Stop-Tunnel
    Write-Host "Tunnel E045 arrete."
    exit 0
}

Stop-Tunnel
Assert-PortAvailable

$remote = @"
set -Eeuo pipefail
cd $RemoteRepository
NS="`${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="`${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
ROOT="`${PROOFTAG_E045_OUTPUT_ROOT:-/data/e045-foundation-v1}"

kubectl scale deployment "`$DEP" -n "`$NS" --replicas=1 >/dev/null
kubectl rollout status deployment/"`$DEP" -n "`$NS" --timeout=1200s >/dev/null
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "/workspace/notebooks/$Notebook"
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- python - "`$ROOT/LATEST.json" <<'PY'
import json, sys
from pathlib import Path
latest=Path(sys.argv[1])
payload=json.loads(latest.read_text(encoding='utf-8'))
assert payload['status']=='complete', payload
plan=Path(payload['plan_dir'])
assert (plan/'COMPLETE.json').is_file(), plan
print('E045_PLAN_DIR='+str(plan))
PY

ENC="`$(kubectl get secret prooftag-qr-notebook -n "`$NS" -o jsonpath='{.data.token}')"
TOKEN="`$(printf '%s' "`$ENC" | base64 --decode)"
IP="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.clusterIP}')"
PORT="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.ports[0].port}')"
printf 'JUPYTER_TOKEN=%s\nJUPYTER_TARGET=%s:%s\n' "`$TOKEN" "`$IP" "`$PORT"
"@

$bytes = [Text.Encoding]::UTF8.GetBytes($remote.Replace("`r`n", "`n"))
$base64 = [Convert]::ToBase64String($bytes)
$output = & ssh $Server "printf '%s' '$base64' | base64 --decode | bash"
if ($LASTEXITCODE -ne 0) {
    throw "Verification distante E045 en echec."
}

$tokenLine = $output | Where-Object { $_ -like "JUPYTER_TOKEN=*" } | Select-Object -Last 1
$targetLine = $output | Where-Object { $_ -like "JUPYTER_TARGET=*" } | Select-Object -Last 1
if (-not $tokenLine -or -not $targetLine) {
    throw "Le serveur n'a pas retourne le token ou la cible Jupyter."
}
$token = $tokenLine.Substring("JUPYTER_TOKEN=".Length)
$target = $targetLine.Substring("JUPYTER_TARGET=".Length)

[System.IO.File]::Delete($logFile)
$arguments = @(
    "-N",
    "-v",
    "-E", $logFile,
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-L", "${LocalPort}:${target}",
    $Server
)
Write-Host "Une fenetre SSH s'ouvre : saisir le mot de passe de $Server."
$tunnel = Start-Process ssh -ArgumentList $arguments -PassThru
Set-Content -LiteralPath $pidFile -Value $tunnel.Id

$url = "http://127.0.0.1:${LocalPort}/lab/tree/notebooks/${Notebook}?token=$token"
$headers = @{ Authorization = "token $token" }
for ($attempt = 0; $attempt -lt 240; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        Invoke-WebRequest "http://127.0.0.1:${LocalPort}/api" `
            -Headers $headers -TimeoutSec 1 | Out-Null
        Start-Process $url
        Write-Host "E045 ouvert : $url"
        Write-Host "Pour fermer : .\scripts\e045-remote.ps1 -Stop"
        exit 0
    }
    catch {
        if ($tunnel.HasExited) {
            throw "Tunnel SSH E045 arrete. Journal : $logFile"
        }
    }
}
throw "Jupyter E045 ne repond pas."
