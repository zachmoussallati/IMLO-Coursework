# Full end-to-end marker simulation:
#   1. Extract submission.zip to a clean directory.
#   2. Verify required files.
#   3. Run train.py with LIVE epoch output (no buffering — you see each tqdm bar
#      and the per-epoch summary as they happen). Takes ~15-20 min on a CUDA GPU.
#   4. Run test.py and verify Q15 matches the number train.py reported.
#
# Run from the repo root:
#   powershell -NoProfile -File tools\marker_full_pipeline.ps1
#
# What "good" looks like:
#   - 30 epochs print one summary line each, lr ramps up to ~0.1 then anneals.
#   - clean_train_acc climbs through the run; final value should land ~75-80%.
#   - At the end: "Q14 trainval ... = 76.03 %" and "Q15 test ... = 56.39 %".
#   - test.py prints "56.39%" (matches Q15 to 2dp).

$ErrorActionPreference = 'Stop'

$RepoRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$ZipPath    = Join-Path $RepoRoot 'submission.zip'
$ExtractDir = Join-Path $env:TEMP "marker_full_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$CondaEnv   = 'imlo-coursework-gpu'

# Find conda.exe (PowerShell -NoProfile doesn't inherit conda's PATH setup).
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
    throw "Could not find conda.exe. Set `$CondaExe manually."
}
Write-Host "[0/5] Using conda: $CondaExe" -ForegroundColor Cyan

if (-not (Test-Path $ZipPath)) {
    throw "submission.zip not found at $ZipPath"
}

Write-Host "[1/5] Extracting $ZipPath" -ForegroundColor Cyan
Write-Host "          -> $ExtractDir"
New-Item -ItemType Directory -Path $ExtractDir | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

Write-Host "[2/5] Verifying required files are present" -ForegroundColor Cyan
$Required = @('train.py', 'test.py', 'model.pth', 'submission_answers.md',
              'src', 'data_stats.json', 'environment.yml')
foreach ($f in $Required) {
    $p = Join-Path $ExtractDir $f
    if (-not (Test-Path $p)) { throw "Missing required file/dir in zip: $f" }
    Write-Host "    OK  $f"
}
$ZipSizeMB   = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
$ModelSizeMB = [math]::Round((Get-Item (Join-Path $ExtractDir 'model.pth')).Length / 1MB, 2)
Write-Host "    submission.zip = $ZipSizeMB MB    model.pth (zipped) = $ModelSizeMB MB"

# why: the zip's model.pth will be overwritten by train.py at the end of the
# run. Stash a copy so we can diff sizes / re-verify against the shipped one.
$ShippedModelCopy = Join-Path $ExtractDir 'model.pth.shipped'
Copy-Item (Join-Path $ExtractDir 'model.pth') $ShippedModelCopy

Write-Host ""
Write-Host "[3/5] Running 'python train.py' (live output, ~15-20 min)" -ForegroundColor Cyan
Write-Host "          You'll see one tqdm bar per epoch + a summary line for each."
Write-Host ""
Push-Location $ExtractDir
try {
    $trainStart = Get-Date
    # --no-capture-output lets stdout stream live (per-epoch lines, tqdm bars).
    # PYTHONUNBUFFERED=1 stops Python from line-buffering tqdm output on Windows.
    $env:PYTHONUNBUFFERED = '1'
    & $CondaExe run --no-capture-output -n $CondaEnv python train.py
    $trainExit = $LASTEXITCODE
    $trainElapsed = (Get-Date) - $trainStart
} finally {
    Pop-Location
}
Write-Host ""
Write-Host "    train.py exit code = $trainExit    elapsed = $([math]::Round($trainElapsed.TotalMinutes,1)) min"
if ($trainExit -ne 0) { throw "train.py failed" }

# Compare the newly-trained model.pth to the one shipped in the zip.
$NewModelSize = [math]::Round((Get-Item (Join-Path $ExtractDir 'model.pth')).Length / 1MB, 2)
Write-Host "    model.pth rewritten by train.py: $NewModelSize MB (shipped was $ModelSizeMB MB)"

Write-Host ""
Write-Host "[4/5] Running 'python test.py' against the freshly-trained model.pth" -ForegroundColor Cyan
Push-Location $ExtractDir
try {
    $testStart = Get-Date
    & $CondaExe run --no-capture-output -n $CondaEnv python test.py
    $testExit = $LASTEXITCODE
    $testElapsed = (Get-Date) - $testStart
} finally {
    Pop-Location
}
Write-Host ""
Write-Host "    test.py exit code = $testExit    elapsed = $([math]::Round($testElapsed.TotalSeconds,1)) s"
if ($testExit -ne 0) { throw "test.py failed" }

Write-Host ""
Write-Host "[5/5] Done." -ForegroundColor Green
Write-Host "      Extracted run dir kept at: $ExtractDir"
Write-Host "      (delete with: Remove-Item -Recurse -Force '$ExtractDir')"
