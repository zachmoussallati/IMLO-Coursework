# Simulate what a marker does with submission.zip:
#   1. Extract the zip to a clean directory.
#   2. Run `python test.py` from inside it using the conda env from environment-gpu.yml.
#   3. Capture the printed accuracy and exit code.
#
# Run from the repo root:
#   powershell -NoProfile -File tools\marker_simulation.ps1
#
# Expected output ends with one line containing "56.39%".

$ErrorActionPreference = 'Stop'

$RepoRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$ZipPath    = Join-Path $RepoRoot 'submission.zip'
$ExtractDir = Join-Path $env:TEMP "marker_check_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$CondaEnv   = 'imlo-coursework-gpu'   # change to 'imlo-coursework' for the CPU env

# Locate conda.exe (PowerShell -NoProfile doesn't inherit conda's PATH setup).
$CondaExe = $null
$Candidates = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:ProgramData\anaconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\Scripts\conda.exe"
)
foreach ($c in $Candidates) {
    if (Test-Path $c) { $CondaExe = $c; break }
}
if (-not $CondaExe) {
    $CondaExe = (Get-Command conda -ErrorAction SilentlyContinue).Source
}
if (-not $CondaExe) {
    throw "Could not find conda.exe in standard install locations. Set `$CondaExe` manually."
}
Write-Host "[0/4] Using conda: $CondaExe"

if (-not (Test-Path $ZipPath)) {
    throw "submission.zip not found at $ZipPath"
}

Write-Host "[1/4] Extracting $ZipPath -> $ExtractDir"
New-Item -ItemType Directory -Path $ExtractDir | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

Write-Host "[2/4] Verifying required files are present"
$Required = @('train.py', 'test.py', 'model.pth', 'submission_answers.md', 'src')
foreach ($f in $Required) {
    $p = Join-Path $ExtractDir $f
    if (-not (Test-Path $p)) {
        throw "Missing required file/dir in zip: $f"
    }
    Write-Host "    OK  $f"
}

$ZipSizeMB  = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
$ModelSizeMB = [math]::Round((Get-Item (Join-Path $ExtractDir 'model.pth')).Length / 1MB, 2)
Write-Host "    submission.zip = $ZipSizeMB MB    model.pth = $ModelSizeMB MB"

Write-Host "[3/4] Running 'conda run -n $CondaEnv python test.py' from $ExtractDir"
Push-Location $ExtractDir
try {
    $start = Get-Date
    & $CondaExe run --no-capture-output -n $CondaEnv python test.py
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "[4/4] Result"
Write-Host "    exit code = $exitCode"
Write-Host "    elapsed   = $([math]::Round($elapsed.TotalSeconds, 1)) s"

if ($exitCode -ne 0) {
    Write-Host "    ERROR: test.py exited non-zero" -ForegroundColor Red
    exit $exitCode
}

Write-Host ""
Write-Host "Cleanup: rm -r $ExtractDir   (left in place so you can inspect)"
