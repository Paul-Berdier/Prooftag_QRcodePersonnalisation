param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18888,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$Notebook = "31_e036_gamma1000_trust_region.ipynb"
$pidFile = Join-Path $env:TEMP "prooftag-qr-e036-tunnel.pid"
$logFile = Join-Path $env:TEMP "prooftag-qr-e036-ssh.log"

function Stop-E036Tunnel {
    if (-not (Test-Path $pidFile)) { return }
    $tunnelPid = [int](Get-Content $pidFile -Raw)
    $process = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -eq "ssh") {
        Stop-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

if ($Stop) {
    Stop-E036Tunnel
    Write-Host "Tunnel E036 arrêté."
    exit 0
}

Stop-E036Tunnel

$remote = @'
cd ~/apps/Prooftag_QRcodePersonnalisation || exit 1
NS="${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
MODE="$(kubectl get deployment "$DEP" -n "$NS" -o jsonpath='{.spec.template.metadata.annotations.prooftag\.io/notebook-mode}')"
if [ "$MODE" != "advisor-cpu" ]; then
  echo "E036 notebook mode invalide: $MODE" >&2
  exit 1
fi
kubectl exec -n "$NS" deployment/"$DEP" -c notebook -- test -f /workspace/notebooks/31_e036_gamma1000_trust_region.ipynb || exit 1
ENC="$(kubectl get secret prooftag-qr-notebook -n "$NS" -o jsonpath='{.data.token}')"
TOKEN="$(printf '%s' "$ENC" | base64 --decode)"
IP="$(kubectl get service "$DEP" -n "$NS" -o jsonpath='{.spec.clusterIP}')"
PORT="$(kubectl get service "$DEP" -n "$NS" -o jsonpath='{.spec.ports[0].port}')"
printf 'JUPYTER_TOKEN=%s\n' "$TOKEN"
printf 'JUPYTER_TARGET=%s:%s\n' "$IP" "$PORT"
'@

$remoteOutput = & ssh $Server $remote
if ($LASTEXITCODE -ne 0) { throw "Notebook E036 distant indisponible." }
$tokenLine = $remoteOutput | Where-Object { $_ -like "JUPYTER_TOKEN=*" } | Select-Object -Last 1
$targetLine = $remoteOutput | Where-Object { $_ -like "JUPYTER_TARGET=*" } | Select-Object -Last 1
if (-not $tokenLine -or -not $targetLine) { throw "Token/cible Jupyter E036 absents." }
$token = $tokenLine.Substring("JUPYTER_TOKEN=".Length)
$target = $targetLine.Substring("JUPYTER_TARGET=".Length)
if ($target -notmatch '^\d{1,3}(\.\d{1,3}){3}:\d+$') { throw "Cible Jupyter invalide: $target" }

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $LocalPort)
try { $listener.Start() } catch { throw "Port $LocalPort occupé. Utilise -LocalPort 18889." } finally { $listener.Stop() }

Remove-Item $logFile -Force -ErrorAction SilentlyContinue
$args = @(
    "-N", "-v", "-E", $logFile,
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-L", "${LocalPort}:${target}",
    $Server
)
Write-Host "Une fenêtre SSH s'ouvre : saisir le mot de passe de $Server."
$tunnel = Start-Process -FilePath "ssh" -ArgumentList $args -WindowStyle Normal -PassThru
Set-Content -LiteralPath $pidFile -Value $tunnel.Id

$url = "http://127.0.0.1:${LocalPort}/lab/tree/notebooks/${Notebook}?token=$token"
$headers = @{ Authorization = "token $token" }
for ($attempt = 0; $attempt -lt 240; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:${LocalPort}/api" -Headers $headers -TimeoutSec 1 | Out-Null
        Start-Process $url
        Write-Host "E036 ouvert : $url"
        Write-Host "Le notebook affiche le parent, E035 paper/upstream et les trois sorties E036."
        exit 0
    }
    catch {
        if ($tunnel.HasExited) { throw "Tunnel SSH E036 arrêté. Log: $logFile" }
    }
}
throw "Jupyter E036 n'a pas répondu sur le port $LocalPort."
