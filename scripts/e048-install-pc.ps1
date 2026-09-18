param(
    [Parameter(Mandatory=$true)][string]$ZipPath,
    [Parameter(Mandatory=$true)][string]$RepoPath
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$ManagedPaths = @(
    "MANIFEST_E048_SRMPGD.json",
    "README_E048_SRMPGD.md",
    "docs/TESTS_E048_SRMPGD.txt",
    "nightops/.dockerignore",
    "nightops/Dockerfile.e048",
    "nightops/e048_config.json",
    "nightops/qrnight/e048_host.py",
    "nightops/qrnight/e048_worker.py",
    "nightops/tests/test_e048.py",
    "scripts/e048-srmpgd.sh",
    "scripts/e048-install-pc.ps1"
)

Set-Location -LiteralPath $RepoPath

if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    throw "ZIP absent: $ZipPath"
}

# A failed V3 install may have left only the E048 overlay dirty. Repair exactly
# those managed files, but never discard an unrelated user modification.
$dirtyLines = @(git status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE) { throw "git status a échoué" }

if ($dirtyLines.Count -gt 0) {
    $dirtyPaths = @()
    foreach ($line in $dirtyLines) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line.Length -lt 4) { continue }
        $path = $line.Substring(3).Trim('"')
        # Rename output can contain "old -> new"; neither form is expected here.
        if ($path -like '* -> *') {
            throw "Repo sale avec renommage inattendu: $path"
        }
        $dirtyPaths += $path.Replace('\\','/')
    }

    $unexpected = @($dirtyPaths | Where-Object { $_ -notin $ManagedPaths })
    if ($unexpected.Count -gt 0) {
        git status --short
        throw "Repo non propre hors overlay E048; aucune modification utilisateur ne sera écrasée: $($unexpected -join ', ')"
    }

    if ($dirtyPaths.Count -gt 0) {
        Write-Host "Réparation de l'overlay E048 précédent resté non commité..."
        git restore --staged --worktree -- $dirtyPaths
        if ($LASTEXITCODE) { throw "Impossible de restaurer l'ancien overlay E048" }
    }
}

if (@(git status --porcelain).Count -gt 0) {
    git status --short
    throw "Repo toujours non propre après réparation E048"
}

git fetch origin
if ($LASTEXITCODE) { throw "git fetch a échoué" }

git switch main
if ($LASTEXITCODE) { throw "git switch a échoué" }

git pull --ff-only origin main
if ($LASTEXITCODE) { throw "git pull a échoué" }

$temp = Join-Path $env:TEMP ("e048-v4-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -LiteralPath $ZipPath -DestinationPath $temp

$manifestPath = Join-Path $temp "MANIFEST_E048_SRMPGD.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

if ($manifest.release_version -ne "1.2.0-e048-e040-canary") {
    throw "Mauvaise release E048"
}
if ($manifest.pack_revision -ne "v4-cross-platform-canary-tests") {
    throw "Mauvaise révision du pack E048: $($manifest.pack_revision)"
}

Write-Host "Vérification SHA-256 du pack..."
foreach ($item in $manifest.files.PSObject.Properties) {
    $relative = [string]$item.Name
    $source = Join-Path $temp $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Fichier absent du ZIP: $relative"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    if ($actual -ne [string]$item.Value) {
        throw "SHA256 invalide: $relative"
    }
}

Write-Host "Intégration E048 V4..."
foreach ($item in $manifest.files.PSObject.Properties) {
    $relative = [string]$item.Name
    $source = Join-Path $temp $relative
    $destination = Join-Path $RepoPath $relative
    $parent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $RepoPath "MANIFEST_E048_SRMPGD.json") -Force

Write-Host "Compilation Python..."
$env:PYTHONPATH = "$RepoPath\nightops"
python -m py_compile nightops\qrnight\e048_host.py nightops\qrnight\e048_worker.py
if ($LASTEXITCODE) { throw "py_compile a échoué" }

Write-Host "Tests E048 V4..."
python -m unittest discover -s nightops\tests -p test_e048.py -v
if ($LASTEXITCODE) { throw "tests E048 échoués" }

# Recheck the copied bytes after tests before staging anything.
foreach ($item in $manifest.files.PSObject.Properties) {
    $relative = [string]$item.Name
    $destination = Join-Path $RepoPath $relative
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    if ($actual -ne [string]$item.Value) {
        throw "Le repo a divergé du pack pendant les tests: $relative"
    }
}

git add -- $ManagedPaths
if ($LASTEXITCODE) { throw "git add a échoué" }

git diff --cached --check
if ($LASTEXITCODE) { throw "git diff --cached --check a échoué" }

Write-Host ""
Write-Host "E048 V4 OK: 41 tests doivent être PASS et l'overlay est indexé pour commit."
Write-Host "Lance maintenant le commit/push indiqué dans la réponse ChatGPT."
