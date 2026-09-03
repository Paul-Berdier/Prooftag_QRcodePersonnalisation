param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18946,
    [switch]$Atlas,
    [switch]$Partial,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$Notebook = if ($Atlas) {
    "49_e046_visual_atlas.ipynb"
}
else {
    "48_e046_controlled_best_generator.ipynb"
}
$pidFile = Join-Path $env:TEMP "prooftag-qr-e046-tunnel.pid"
$logFile = Join-Path $env:TEMP "prooftag-qr-e046-ssh.log"

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
    Write-Host "Tunnel E046 arrete."
    exit 0
}

Stop-Tunnel
Assert-PortAvailable

$partialFlag = if ($Partial) { "1" } else { "0" }
$remote = @"
set -Eeuo pipefail
cd $RemoteRepository
NS="`${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="`${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
ROOT="`${PROOFTAG_E046_OUTPUT_ROOT:-/data/e046-controlled-best-generator-v1}"
PARTIAL="$partialFlag"

ACTIVE="`$(kubectl get pods -n "`$NS"   -l prooftag.io/experiment=e046-controlled-best-generator-v1   --field-selector=status.phase=Running   -o name 2>/dev/null || true)"
if [ -n "`$ACTIVE" ]; then
  echo "Un Job GPU E046 est actif : notebook non demarre pour ne pas concurrencer la RTX." >&2
  echo "`$ACTIVE" >&2
  exit 1
fi

kubectl scale deployment "`$DEP" -n "`$NS" --replicas=1 >/dev/null
kubectl rollout status deployment/"`$DEP" -n "`$NS" --timeout=1200s >/dev/null
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- \
  test -f "/workspace/notebooks/$Notebook"

kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- \
  python - "`$ROOT/LATEST.json" "`$PARTIAL" <<'PY'
import json, sys
from pathlib import Path

latest = Path(sys.argv[1])
partial = sys.argv[2] == "1"
payload = json.loads(latest.read_text(encoding="utf-8"))
plan = Path(payload["plan_dir"])
assert (plan / "plan.json").is_file(), plan
if not partial:
    assert payload["status"] == "complete", payload
    assert (plan / "verdict.json").is_file(), plan
print("E046_PLAN_DIR=" + str(plan))
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
    throw "Verification distante E046 en echec."
}

$tokenLine = $output |
    Where-Object { $_ -like "JUPYTER_TOKEN=*" } |
    Select-Object -Last 1
$targetLine = $output |
    Where-Object { $_ -like "JUPYTER_TARGET=*" } |
    Select-Object -Last 1
if (-not $tokenLine -or -not $targetLine) {
    throw "Token ou cible Jupyter E046 absent."
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
        Invoke-WebRequest `
            "http://127.0.0.1:${LocalPort}/api" `
            -Headers $headers `
            -TimeoutSec 1 | Out-Null
        Start-Process $url
        Write-Host "E046 ouvert : $url"
        Write-Host "Pour fermer : .\scripts\e046-remote.ps1 -Stop"
        exit 0
    }
    catch {
        if ($tunnel.HasExited) {
            throw "Tunnel SSH E046 arrete. Journal : $logFile"
        }
    }
}
throw "Jupyter E046 ne repond pas."
