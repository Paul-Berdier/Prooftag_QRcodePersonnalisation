param(
    [string]$Server = "paul@pcIA",
    [int]$LocalPort = 18080,
    [int]$RemotePort = 18080,
    [string]$Namespace = "qr-core",
    [string]$Service = "prooftag-qr-svc",
    [switch]$NoBrowser,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$stateFile = Join-Path $env:TEMP "prooftag-qr-lab-tunnel.json"
$sshLogFile = Join-Path $env:TEMP "prooftag-qr-lab-ssh.log"

function Stop-LabTunnel {
    if (-not [System.IO.File]::Exists($stateFile)) {
        return $false
    }
    try {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -eq "ssh") {
            Stop-Process -Id $process.Id
        }
    }
    finally {
        [System.IO.File]::Delete($stateFile)
    }
    return $true
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
        throw "Le port local $LocalPort est deja occupe. Utiliser par exemple -LocalPort 18081."
    }
    finally {
        $listener.Stop()
    }
}

if ($Stop) {
    $stopped = Stop-LabTunnel
    if ($stopped) {
        Write-Host "Tunnel du laboratoire arrete."
    }
    else {
        Write-Host "Aucun tunnel du laboratoire n'etait enregistre."
    }
    exit 0
}

Stop-LabTunnel | Out-Null
Assert-LocalPortAvailable
[System.IO.File]::Delete($sshLogFile)

$remoteCommand = (
    "kubectl port-forward -n {0} service/{1} {2}:8080 --address 127.0.0.1" -f
    $Namespace, $Service, $RemotePort
)
$arguments = @(
    "-v",
    "-E", $sshLogFile,
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", "${LocalPort}:127.0.0.1:${RemotePort}",
    $Server,
    $remoteCommand
)

$tunnel = $null
try {
    Write-Host "Une fenetre SSH va s'ouvrir : saisir le mot de passe de $Server."
    $tunnel = Start-Process `
        -FilePath "ssh" `
        -ArgumentList $arguments `
        -WindowStyle Normal `
        -PassThru
    @{
        pid = $tunnel.Id
        server = $Server
        local_port = $LocalPort
        remote_port = $RemotePort
        started_at = [DateTimeOffset]::UtcNow.ToString("O")
    } | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8

    $url = "http://127.0.0.1:${LocalPort}/lab"
    for ($attempt = 0; $attempt -lt 240; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($tunnel.HasExited) {
            throw "Le tunnel SSH s'est arrete. Journal : $sshLogFile"
        }
        try {
            $health = Invoke-RestMethod `
                -Uri "http://127.0.0.1:${LocalPort}/healthz" `
                -TimeoutSec 1
            if ($health.status -eq "ok") {
                if (-not $NoBrowser) {
                    Start-Process $url
                }
                Write-Host "Laboratoire Prooftag ouvert : $url"
                Write-Host "Pour fermer le tunnel : .\scripts\lab-remote.ps1 -Stop"
                exit 0
            }
        }
        catch {
            if ($tunnel.HasExited) {
                throw "Le tunnel SSH s'est arrete. Journal : $sshLogFile"
            }
        }
    }
    throw "L'API Prooftag n'a pas repondu sur le port local $LocalPort."
}
catch {
    $originalError = $_
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
    [System.IO.File]::Delete($stateFile)
    throw $originalError
}
