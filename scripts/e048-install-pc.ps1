param(
    [Parameter(Mandatory=$true)][string]$ZipPath,
    [Parameter(Mandatory=$true)][string]$RepoPath
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoPath
if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) { throw "ZIP absent: $ZipPath" }
if (@(git status --porcelain).Count -gt 0) { git status --short; throw "Repo non propre" }
git fetch origin
if ($LASTEXITCODE) { throw "git fetch a échoué" }
git switch main
if ($LASTEXITCODE) { throw "git switch a échoué" }
git pull --ff-only origin main
if ($LASTEXITCODE) { throw "git pull a échoué" }
$temp = Join-Path $env:TEMP ("e048-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -LiteralPath $ZipPath -DestinationPath $temp
$manifest = Get-Content -LiteralPath (Join-Path $temp "MANIFEST_E048_SRMPGD.json") -Raw | ConvertFrom-Json
if ($manifest.release_version -ne "1.1.0-e048-canary") { throw "Mauvaise release E048" }
foreach ($item in $manifest.files.PSObject.Properties) {
    $relative = [string]$item.Name
    $source = Join-Path $temp $relative
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    if ($actual -ne [string]$item.Value) { throw "SHA256 invalide: $relative" }
    $destination = Join-Path $RepoPath $relative
    $parent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
Copy-Item -LiteralPath (Join-Path $temp "MANIFEST_E048_SRMPGD.json") -Destination (Join-Path $RepoPath "MANIFEST_E048_SRMPGD.json") -Force
Write-Host "Overlay E048 intégré. Vérification Python..."
python -m py_compile nightops\qrnight\e048_host.py nightops\qrnight\e048_worker.py
if ($LASTEXITCODE) { throw "py_compile a échoué" }
python -m unittest discover -s nightops\tests -p test_e048.py -v
if ($LASTEXITCODE) { throw "tests E048 échoués" }
git add MANIFEST_E048_SRMPGD.json README_E048_SRMPGD.md docs\TESTS_E048_SRMPGD.txt nightops\.dockerignore nightops\Dockerfile.e048 nightops\e048_config.json nightops\qrnight\e048_host.py nightops\qrnight\e048_worker.py nightops\tests\test_e048.py scripts\e048-srmpgd.sh scripts\e048-install-pc.ps1
Write-Host "Overlay prêt pour commit. Lancez git diff --cached --check puis commit/push."
