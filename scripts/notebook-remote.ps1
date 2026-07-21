param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18888,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $env:TEMP "prooftag-qr-notebook-tunnel.pid"

function Stop-LocalTunnel {
    if (-not (Test-Path $pidFile)) {
        return
    }
    $tunnelPid = [int](Get-Content $pidFile -Raw)
    $process = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -eq "ssh") {
        Stop-Process -Id $tunnelPid
    }
    Remove-Item -LiteralPath $pidFile -Force
}

function Stop-RemoteNotebook {
    & ssh $Server "cd $RemoteRepository && bash scripts/notebook-server.sh stop"
    if ($LASTEXITCODE -ne 0) {
        throw "Impossible d'arreter le notebook distant."
    }
}

function Assert-LocalPortAvailable {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $LocalPort
    )
    try {
        $listener.Start()
    }
    catch {
        throw "Le port local $LocalPort est deja occupe. Utiliser par exemple -LocalPort 18889."
    }
    finally {
        $listener.Stop()
    }
}

if ($Stop) {
    Stop-LocalTunnel
    Stop-RemoteNotebook
    Write-Host "Notebook arrete et charge GPU precedente restauree."
    exit 0
}

Assert-LocalPortAvailable
$remoteStarted = $false
$tunnel = $null
try {
    $remoteOutput = & ssh $Server "cd $RemoteRepository && bash scripts/notebook-server.sh start"
    if ($LASTEXITCODE -ne 0) {
        throw "Impossible de demarrer le notebook distant."
    }
    $remoteStarted = $true
    $tokenLine = $remoteOutput | Where-Object { $_ -like "JUPYTER_TOKEN=*" } |
        Select-Object -Last 1
    if (-not $tokenLine) {
        throw "Le serveur n'a pas retourne le jeton Jupyter."
    }
    $token = $tokenLine.Substring("JUPYTER_TOKEN=".Length)

    Stop-LocalTunnel
    $forwardCommand = "kubectl port-forward -n qr-core service/prooftag-qr-notebook 18888:8888"
    $arguments = @(
        "-N",
        "-L", "${LocalPort}:127.0.0.1:18888",
        $Server,
        $forwardCommand
    )
    $tunnel = Start-Process -FilePath "ssh" -ArgumentList $arguments -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $pidFile -Value $tunnel.Id

    $url = "http://127.0.0.1:${LocalPort}/lab/tree/notebooks/02_generate_live_on_gpu.ipynb?token=$token"
    $headers = @{ Authorization = "token $token" }
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:${LocalPort}/api" `
                -Headers $headers -TimeoutSec 1 | Out-Null
            Start-Process $url
            Write-Host "Notebook GPU ouvert sur le PC : $url"
            Write-Host "Pour arreter et restaurer le GPU : .\scripts\notebook-remote.ps1 -Stop"
            exit 0
        }
        catch {
            if ($tunnel.HasExited) {
                throw "Le tunnel SSH s'est arrete avant l'ouverture de Jupyter."
            }
        }
    }
    throw "Jupyter n'a pas repondu sur le port local $LocalPort."
}
catch {
    $originalError = $_
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    if ($remoteStarted) {
        try {
            Stop-RemoteNotebook
        }
        catch {
            Write-Warning "La restauration distante a aussi echoue : $($_.Exception.Message)"
        }
    }
    throw $originalError
}
