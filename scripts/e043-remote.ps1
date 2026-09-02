param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18943,
    [switch]$Pipeline,
    [switch]$Stop
)
$ErrorActionPreference = "Stop"
$Notebook = if ($Pipeline) { "42_e043_final_pipeline_visualizer.ipynb" } else { "41_e043_scanner_cell_frontier.ipynb" }
$pidFile = Join-Path $env:TEMP "prooftag-qr-e043-tunnel.pid"
function Stop-Tunnel { if (Test-Path $pidFile) { $id=[int](Get-Content $pidFile -Raw); Stop-Process -Id $id -ErrorAction SilentlyContinue; Remove-Item $pidFile -Force -ErrorAction SilentlyContinue } }
if ($Stop) { Stop-Tunnel; Write-Host "Tunnel E043 arrete."; exit 0 }
Stop-Tunnel
$listener=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$LocalPort); try{$listener.Start()}catch{throw "Port $LocalPort occupe. Utilise -LocalPort $($LocalPort+1)."}finally{$listener.Stop()}
$remote=@"
set -Eeuo pipefail
cd $RemoteRepository
NS="`${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="`${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
RESULTS="`${PROOFTAG_E043_RESULTS_DIR:-/data/e043-scanner-cell-frontier-v1}"
kubectl scale deployment "`$DEP" -n "`$NS" --replicas=1 >/dev/null
kubectl rollout status deployment/"`$DEP" -n "`$NS" --timeout=1200s >/dev/null
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "`$RESULTS/verdict.json" || { echo "E043 verdict absent dans le PVC : `$RESULTS/verdict.json" >&2; exit 1; }
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "/workspace/notebooks/$Notebook"
ENC="`$(kubectl get secret prooftag-qr-notebook -n "`$NS" -o jsonpath='{.data.token}')"
TOKEN="`$(printf '%s' "`$ENC" | base64 --decode)"
IP="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.clusterIP}')"
PORT="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.ports[0].port}')"
printf 'JUPYTER_TOKEN=%s\nJUPYTER_TARGET=%s:%s\n' "`$TOKEN" "`$IP" "`$PORT"
"@
$bytes=[Text.Encoding]::UTF8.GetBytes($remote.Replace("`r`n","`n")); $b64=[Convert]::ToBase64String($bytes)
$out=& ssh $Server "printf '%s' '$b64' | base64 --decode | bash"; if($LASTEXITCODE -ne 0){throw "Verification distante E043 en echec."}
$token=($out|Where-Object{$_ -like 'JUPYTER_TOKEN=*'}|Select-Object -Last 1).Substring(14)
$target=($out|Where-Object{$_ -like 'JUPYTER_TARGET=*'}|Select-Object -Last 1).Substring(15)
$args=@('-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-L',"${LocalPort}:${target}",$Server)
Write-Host "Une fenetre SSH s'ouvre : saisir le mot de passe de $Server."
$p=Start-Process ssh -ArgumentList $args -PassThru; Set-Content $pidFile $p.Id
$url="http://127.0.0.1:${LocalPort}/lab/tree/notebooks/${Notebook}?token=$token"
for($i=0;$i -lt 240;$i++){Start-Sleep -Milliseconds 500; try{Invoke-WebRequest "http://127.0.0.1:${LocalPort}/api" -Headers @{Authorization="token $token"} -TimeoutSec 1|Out-Null; Start-Process $url; Write-Host "E043 ouvert : $url"; exit 0}catch{if($p.HasExited){throw "Tunnel SSH E043 arrete."}}}
throw "Jupyter E043 ne repond pas."
