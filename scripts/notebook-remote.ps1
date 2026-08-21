param(
    [string]$Server = "paul@pcIA",
    [string]$RemoteRepository = "~/apps/Prooftag_QRcodePersonnalisation",
    [int]$LocalPort = 18888,
    [ValidateSet(
        "02_generate_live_on_gpu.ipynb",
        "03_srpg_parameter_search.ipynb",
        "04_e007_contextual_optimizer.ipynb",
        "05_controlnet_model_bakeoff.ipynb",
        "06_nacholmo_generate_live.ipynb",
        "07_diffqrcoder_official_live.ipynb",
        "08_diffqrcoder_vs_qrbtf_four_prompts.ipynb",
        "09_diffqrcoder_faithful_srmpgd.ipynb",
        "10_exact_geometry_sd15_sd21_policy.ipynb",
        "11_e014a_qart_blueprint_bakeoff.ipynb",
        "12_e014b_freeqr_latent_fusion.ipynb",
        "13_e015_aesthetic_backbone_reference.ipynb",
        "14_e016_differentiable_scan_surrogate.ipynb",
        "15_e014c_stage2_determinism_diagnostic.ipynb",
        "16_e014b_statistical_freeqr_confirmation.ipynb",
        "17_e014b_multicontext_generalization.ipynb",
        "18_e014d_functional_late_rediffusion.ipynb",
        "19_e014e_mechanism_window_ablation.ipynb",
        "20_e014f_unseen_generalization_cascade.ipynb",
        "21_e026_prompt_parameter_advisor.ipynb",
        "22_e027_srmpgd_policy_holdout.ipynb",
        "23_e028_hierarchical_prompt_advisor.ipynb",
        "24_e029_srmpgd_exact_raster_recovery.ipynb",
        "25_e030_reliable_qrverify_cascade.ipynb",
        "26_e031_prospective_stage2_holdout.ipynb"
    )]
    [string]$Notebook = "02_generate_live_on_gpu.ipynb",
    [switch]$Reset,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $env:TEMP "prooftag-qr-notebook-tunnel.pid"
$tunnelLogFile = Join-Path $env:TEMP "prooftag-qr-notebook-ssh.log"

function Stop-LocalTunnel {
    if (-not (Test-Path $pidFile)) {
        return
    }
    $tunnelPid = [int](Get-Content $pidFile -Raw)
    $process = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -eq "ssh") {
        Stop-Process -Id $tunnelPid
    }
    [System.IO.File]::Delete($pidFile)
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

if ($Stop -and $Reset) {
    throw "Utiliser soit -Stop, soit -Reset, pas les deux."
}

if ($Stop) {
    Stop-LocalTunnel
    Stop-RemoteNotebook
    Write-Host "Notebook arrete et charge GPU precedente restauree."
    exit 0
}

Stop-LocalTunnel
Assert-LocalPortAvailable
$remoteStarted = $false
$tunnel = $null
try {
    $remoteAction = if ($Reset) { "reset" } else { "start" }
    $remoteOutput = & ssh $Server "cd $RemoteRepository && bash scripts/notebook-server.sh $remoteAction $Notebook"
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
    $targetLine = $remoteOutput | Where-Object { $_ -like "JUPYTER_TARGET=*" } |
        Select-Object -Last 1
    if (-not $targetLine) {
        throw "Le serveur n'a pas retourne la cible reseau de Jupyter."
    }
    $target = $targetLine.Substring("JUPYTER_TARGET=".Length)
    if ($target -notmatch '^\d{1,3}(\.\d{1,3}){3}:\d+$') {
        throw "Cible reseau Jupyter invalide : $target"
    }

    [System.IO.File]::Delete($tunnelLogFile)
    $arguments = @(
        "-N",
        "-v",
        "-E", $tunnelLogFile,
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
            Invoke-WebRequest -Uri "http://127.0.0.1:${LocalPort}/api" `
                -Headers $headers -TimeoutSec 1 | Out-Null
            Start-Process $url
            Write-Host "Notebook ouvert sur le PC : $url"
            if ($Notebook -in @(
                "21_e026_prompt_parameter_advisor.ipynb",
                "22_e027_srmpgd_policy_holdout.ipynb",
                "23_e028_hierarchical_prompt_advisor.ipynb",
                "24_e029_srmpgd_exact_raster_recovery.ipynb",
                "26_e031_prospective_stage2_holdout.ipynb"
            )) {
                Write-Host "Mode campagne : Jupyter reste sur CPU et l'API conserve la RTX pour generer les donnees."
            }
            elseif ($Notebook -eq "25_e030_reliable_qrverify_cascade.ipynb") {
                Write-Host "Mode analyse hors ligne : CPU uniquement, aucune generation et aucune charge GPU modifiee."
            }
            else {
                Write-Host "Mode generation directe : ce notebook utilise la RTX."
            }
            if ($Reset) {
                Write-Host "Tous les anciens kernels ont ete supprimes et la VRAM a ete liberee."
            }
            Write-Host "Pour arreter et restaurer le GPU : .\scripts\notebook-remote.ps1 -Stop"
            exit 0
        }
        catch {
            if ($tunnel.HasExited) {
                throw "Le tunnel SSH s'est arrete avant l'ouverture de Jupyter. Journal : $tunnelLogFile"
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
    [System.IO.File]::Delete($pidFile)
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
