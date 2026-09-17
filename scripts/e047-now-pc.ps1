# Execute on Windows, inside the user's PowerShell terminal. No server connection here.
& {
    $ErrorActionPreference = "Stop"
    $Repo = "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
    $Zip = Join-Path $env:USERPROFILE "Downloads\Prooftag_QR_E047_START_NOW_v2.zip"

    function Git-OK {
        & git @args
        if ($LASTEXITCODE -ne 0) { throw "Git a echoue : git $args" }
    }

    if (-not (Test-Path -LiteralPath $Repo -PathType Container)) { throw "Repo absent : $Repo" }
    if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) { throw "ZIP absent : $Zip" }
    Set-Location -LiteralPath $Repo
    $Dirty = @(Git-OK status --porcelain)
    if ($Dirty.Count -gt 0) {
        $Dirty | Write-Host
        throw "Le clone PC contient des modifications. Ne pas les ecraser; sauvegarder/commiter avant ce bloc."
    }
    Git-OK fetch origin
    Git-OK switch main
    Git-OK pull --ff-only origin main

    $Temp = Join-Path $env:TEMP ("e047-now-" + [guid]::NewGuid().ToString("N"))
    Expand-Archive -LiteralPath $Zip -DestinationPath $Temp
    $Manifest = Get-Content -LiteralPath (Join-Path $Temp "MANIFEST_NUIT_E047.json") -Raw | ConvertFrom-Json
    if ($Manifest.release_version -ne "1.1.0-immediate") { throw "Mauvaise version du ZIP" }

    foreach ($Item in $Manifest.files.PSObject.Properties) {
        $Relative = [string]$Item.Name
        if ([IO.Path]::IsPathRooted($Relative) -or $Relative.Split('/') -contains '..') {
            throw "Chemin interdit dans le manifeste : $Relative"
        }
        $Source = Join-Path $Temp $Relative
        $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLowerInvariant()
        if ($Actual -ne $Item.Value) { throw "Empreinte invalide : $Relative" }
    }
    $Files = @($Manifest.files.PSObject.Properties.Name) + @("MANIFEST_NUIT_E047.json")
    foreach ($Relative in $Files) {
        $Destination = Join-Path $Repo $Relative
        $Parent = Split-Path -Parent $Destination
        if (-not (Test-Path -LiteralPath $Parent)) { New-Item -ItemType Directory -Path $Parent | Out-Null }
        Copy-Item -LiteralPath (Join-Path $Temp $Relative) -Destination $Destination -Force
    }

    Git-OK add -- @Files
    Git-OK diff --cached --check
    Git-OK diff --cached --stat
    & git diff --cached --quiet
    $DiffCode = $LASTEXITCODE
    if ($DiffCode -eq 1) {
        Git-OK commit -m "fix(e047): immediate start, validated python3 health and explicit run failures"
    } elseif ($DiffCode -ne 0) {
        throw "Impossible de verifier les changements prepares"
    }
    Git-OK push origin main
    Write-Host "COMMIT POUSSE :"
    Git-OK rev-parse HEAD
    Write-Host "PC termine. Executer maintenant le bloc serveur."
}
