param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18944,
    [switch]$Atlas,
    [switch]$Stop
)
$ErrorActionPreference = "Stop"
$Notebook = if ($Atlas) { "46_e044_visual_atlas.ipynb" } else { "45_e044_multiprompt_complete_audit.ipynb" }
$pidFile = Join-Path $env:TEMP "prooftag-qr-e044-tunnel.pid"
function Stop-Tunnel { if (Test-Path $pidFile) { $id=[int](Get-Content $pidFile -Raw); Stop-Process -Id $id -ErrorAction SilentlyContinue; Remove-Item $pidFile -Force -ErrorAction SilentlyContinue } }
if ($Stop) { Stop-Tunnel; Write-Host "Tunnel E044 arrete."; exit 0 }
Stop-Tunnel
$remote=@"
set -Eeuo pipefail
cd $RemoteRepository
NS="`${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="`${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
ROOT="`${PROOFTAG_E044_RESULTS_ROOT:-/data/e044-multi-prompt-best-pipeline-v1}"
kubectl scale deployment "`$DEP" -n "`$NS" --replicas=1 >/dev/null
kubectl rollout status deployment/"`$DEP" -n "`$NS" --timeout=1200s >/dev/null
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "`$ROOT/verdict.json"
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "/workspace/notebooks/$Notebook"
ENC="`$(kubectl get secret prooftag-qr-notebook -n "`$NS" -o jsonpath='{.data.token}')"
TOKEN="`$(printf '%s' "`$ENC" | base64 --decode)"
IP="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.clusterIP}')"
PORT="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.ports[0].port}')"
printf 'JUPYTER_TOKEN=%s\nJUPYTER_TARGET=%s:%s\n' "`$TOKEN" "`$IP" "`$PORT"
"@
$bytes=[Text.Encoding]::UTF8.GetBytes($remote.Replace("`r`n","`n")); $b64=[Convert]::ToBase64String($bytes)
$out=& ssh $Server "printf '%s' '$b64' | base64 --decode | bash"; if($LASTEXITCODE -ne 0){throw "Verification distante E044 en echec."}
$token=($out|Where-Object{$_ -like 'JUPYTER_TOKEN=*'}|Select-Object -Last 1).Substring(14); $target=($out|Where-Object{$_ -like 'JUPYTER_TARGET=*'}|Select-Object -Last 1).Substring(15)
$args=@('-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-L',"${LocalPort}:${target}",$Server)
$p=Start-Process ssh -ArgumentList $args -PassThru; Set-Content $pidFile $p.Id
$url="http://127.0.0.1:${LocalPort}/lab/tree/notebooks/${Notebook}?token=$token"
for($i=0;$i -lt 240;$i++){ Start-Sleep -Milliseconds 500; try{ Invoke-WebRequest "http://127.0.0.1:${LocalPort}/api" -Headers @{Authorization="token $token"} -TimeoutSec 1|Out-Null; Start-Process $url; Write-Host "E044 ouvert : $url"; exit 0 }catch{ if($p.HasExited){throw "Tunnel SSH E044 arrete."} } }
throw "Jupyter E044 ne repond pas."
