param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18888,
    [switch]$Pipeline,
    [switch]$Stop
)
$ErrorActionPreference = "Stop"
$Notebook = if ($Pipeline) { "36_final_qr_pipeline_visualizer.ipynb" } else { "35_e040_srmpgd_checkpoint_frontier.ipynb" }
$pidFile = Join-Path $env:TEMP "prooftag-qr-e040-tunnel.pid"
function Stop-Tunnel { if (Test-Path $pidFile) { $id=[int](Get-Content $pidFile -Raw); Stop-Process -Id $id -ErrorAction SilentlyContinue; Remove-Item $pidFile -Force -ErrorAction SilentlyContinue } }
if ($Stop) { Stop-Tunnel; Write-Host "Tunnel E040 arrete."; exit 0 }
Stop-Tunnel
$listener=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$LocalPort); try{$listener.Start()}catch{throw "Port $LocalPort occupe. Utilise -LocalPort 18889."}finally{$listener.Stop()}
$remote=@"
set -Eeuo pipefail
cd $RemoteRepository
NS="`${PROOFTAG_QR_NAMESPACE:-qr-core}"
DEP="`${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
RESULTS="`${PROOFTAG_E040_RESULTS_DIR:-/data/e040-srmpgd-checkpoint-frontier-v1}"
test -f "`$RESULTS/verdict.json" || { echo "E040 pas termine. Lancer bash scripts/run-e040-checkpoint-frontier.sh" >&2; exit 1; }
kubectl exec -n "`$NS" deployment/"`$DEP" -c notebook -- test -f "/workspace/notebooks/$Notebook"
ENC="`$(kubectl get secret prooftag-qr-notebook -n "`$NS" -o jsonpath='{.data.token}')"
TOKEN="`$(printf '%s' "`$ENC" | base64 --decode)"
IP="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.clusterIP}')"
PORT="`$(kubectl get service "`$DEP" -n "`$NS" -o jsonpath='{.spec.ports[0].port}')"
printf 'JUPYTER_TOKEN=%s\nJUPYTER_TARGET=%s:%s\n' "`$TOKEN" "`$IP" "`$PORT"
"@
$bytes=[Text.Encoding]::UTF8.GetBytes($remote.Replace("`r`n","`n")); $b64=[Convert]::ToBase64String($bytes)
$out=& ssh $Server "printf '%s' '$b64' | base64 --decode | bash"; if($LASTEXITCODE -ne 0){throw "Verification distante E040 en echec."}
$token=($out|Where-Object{$_ -like 'JUPYTER_TOKEN=*'}|Select-Object -Last 1).Substring(14)
$target=($out|Where-Object{$_ -like 'JUPYTER_TARGET=*'}|Select-Object -Last 1).Substring(15)
$args=@('-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-L',"${LocalPort}:${target}",$Server)
Write-Host "Une fenetre SSH s'ouvre : saisir le mot de passe de $Server."
$p=Start-Process ssh -ArgumentList $args -PassThru; Set-Content $pidFile $p.Id
$url="http://127.0.0.1:${LocalPort}/lab/tree/notebooks/${Notebook}?token=$token"
for($i=0;$i -lt 240;$i++){Start-Sleep -Milliseconds 500; try{Invoke-WebRequest "http://127.0.0.1:${LocalPort}/api" -Headers @{Authorization="token $token"} -TimeoutSec 1|Out-Null; Start-Process $url; Write-Host "E040 ouvert : $url"; exit 0}catch{if($p.HasExited){throw "Tunnel SSH E040 arrete."}}}
throw "Jupyter E040 ne repond pas."
