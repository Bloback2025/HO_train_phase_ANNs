<#
Edit-FileSafely.ps1
Atomic, auditable file edit helper.

Usage examples:
  # Dry-run with a payload file
  .\Edit-FileSafely.ps1 -TargetPath .\deterministic_inference.py -PayloadFile .\payload.py -DryRun

  # Apply with payload file, non-interactive (CI)
  .\Edit-FileSafely.ps1 -TargetPath .\deterministic_inference.py -PayloadFile .\payload.py -PyVenvExe .\.venv\Scripts\python.exe -Auto

  # Apply using an inline here-doc (PowerShell string)
  $doc = @'
  # replacement content...
  '@
  .\Edit-FileSafely.ps1 -TargetPath .\deterministic_inference.py -PayloadHereDoc $doc -Auto

Notes:
  - Writes temporary artifacts into edits/ (gitignored recommended).
  - On failure, backup and temp are preserved for inspection.
  - Ledger: edits/edit_ledger.csv appended with extra metadata.
#>

param(
  [Parameter(Mandatory=$true)][string]$TargetPath,
  [Parameter(Mandatory=$false)][string]$PayloadFile,
  [Parameter(Mandatory=$false)][string]$PayloadHereDoc,
  [switch]$DryRun,
  [switch]$Auto,
  [string]$PyVenvExe = ".\.venv\Scripts\python.exe",
  [string]$SmokeModule = "",
  [string]$SmokeFunc = "",
  [string]$SmokeArgs = "",
  [int]$SmokeTimeoutSeconds = 10
)

set -o nounset

function Get-SHA256($path) {
  if (-not (Test-Path $path)) { return $null }
  return (Get-FileHash -Path $path -Algorithm SHA256).Hash
}

function Write-Json($obj, $path) {
  $j = $obj | ConvertTo-Json -Depth 5
  $j | Set-Content -LiteralPath $path -Encoding UTF8
}

# Normalize paths and ensure edits dir
$Target = Resolve-Path -LiteralPath $TargetPath
$RepoRoot = Split-Path -Path $Target -Parent
$EditsDir = Join-Path $RepoRoot "edits"
if (-not (Test-Path $EditsDir)) { New-Item -ItemType Directory -Path $EditsDir | Out-Null }

$ts = (Get-Date).ToString("yyyyMMddHHmmss")
$Backup = "$Target.bak.$ts"
$Preapply = "$Target.preapply.$ts"
$Temp = Join-Path $EditsDir ("temp.edit.$ts.py")
$PyCompileErr = Join-Path $EditsDir ("py_compile_err.$ts.log")
$SmokeOut = Join-Path $EditsDir ("smoke_out.$ts.log")
$Ledger = Join-Path $EditsDir "edit_ledger.csv"

# Create backup (atomic copy)
Copy-Item -LiteralPath $Target -Destination $Backup -Force
$OrigSha = Get-SHA256 $Backup
Write-Output "BACKUP_CREATED,$Backup,$OrigSha"

# Prepare payload into temp (UTF8 no BOM)
if ($PayloadFile) {
  if (-not (Test-Path $PayloadFile)) { Write-Error "PayloadFile not found: $PayloadFile"; exit 2 }
  Copy-Item -LiteralPath $PayloadFile -Destination $Temp -Force
} elseif ($PayloadHereDoc) {
  # Trim consistent leading/trailing newline to avoid accidental indent shifts
  $payloadTrimmed = $PayloadHereDoc.Trim("`r", "`n")
  $payloadTrimmed | Set-Content -LiteralPath $Temp -Encoding UTF8
} else {
  Write-Error "Either PayloadFile or PayloadHereDoc must be provided"
  exit 2
}

# Validate non-empty payload
if ((Get-Item -LiteralPath $Temp).Length -eq 0) {
  Write-Error "Payload is empty: $Temp"
  exit 2
}

# Dry-run: stop after writing temp
if ($DryRun) {
  Write-Output "DRY_RUN: temp payload written to $Temp"
  Write-Output "Temp SHA256: $(Get-SHA256 $Temp)"
  exit 0
}

# Determine python for compile/smoke (prefer provided venv)
$PyExe = if (Test-Path $PyVenvExe) { Resolve-Path -LiteralPath $PyVenvExe } else { "python" }

# py_compile check, capture stderr/stdout to file for diagnostics
$pyArgs = @("-u","-m","py_compile",$Temp)
$procInfo = @{
  FilePath = $PyExe
  ArgumentList = $pyArgs
  NoNewWindow = $true
  RedirectStandardOutput = $true
  RedirectStandardError = $true
  UseNewEnvironment = $false
}
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo.FileName = $PyExe
$proc.StartInfo.Arguments = $pyArgs -join ' '
$proc.StartInfo.RedirectStandardOutput = $true
$proc.StartInfo.RedirectStandardError = $true
$proc.StartInfo.UseShellExecute = $false
$proc.Start() | Out-Null
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
$stdout | Out-File -LiteralPath $PyCompileErr -Encoding UTF8
$stderr | Out-File -LiteralPath $PyCompileErr -Append -Encoding UTF8
if ($proc.ExitCode -ne 0) {
  Write-Output "PY_COMPILE_FAILED, exit=$($proc.ExitCode), log=$PyCompileErr"
  Write-Output "Temp retained at: $Temp ; Backup at: $Backup"
  exit 3
}
Write-Output "PY_COMPILE_OK, log=$PyCompileErr"

# Optional smoke-run (bounded)
if ($SmokeModule -and $SmokeFunc) {
  $smokePy = @"
import importlib, json, sys, traceback
m = importlib.import_module('$SmokeModule')
func = getattr(m, '$SmokeFunc', None)
if func is None:
    print('SMOKE_MISSING_FUNC')
    sys.exit(4)
try:
    # Attempt to call with provided args if given as a Python tuple/list literal
    args = $SmokeArgs
    # If SmokeArgs is empty string, call with no args
    if '$SmokeArgs' == '':
        res = func()
    else:
        # Evaluate literal tuple/list safely by relying on Python eval of simple literal
        res = func(*eval('$SmokeArgs'))
    print('SMOKE_OK', res)
except Exception:
    traceback.print_exc()
    sys.exit(5)
"@

  # Run smoke with timeout
  $start = [System.Diagnostics.Process]::Start($PyExe, "-u")
  $start.StandardInput.WriteLine($smokePy)
  $start.StandardInput.Close()
  $finished = $start.WaitForExit($SmokeTimeoutSeconds * 1000)
  if (-not $finished) {
    try { $start.Kill() } catch {}
    Write-Output "SMOKE_TIMEOUT,$SmokeTimeoutSeconds sec (killed)"
    Write-Output "Temp retained at: $Temp ; Backup at: $Backup"
    exit 5
  }
  $smOut = $start.StandardOutput.ReadToEnd()
  $smErr = $start.StandardError.ReadToEnd()
  $smOut + "`n" + $smErr | Out-File -LiteralPath $SmokeOut -Encoding UTF8
  if ($start.ExitCode -ne 0) {
    Write-Output "SMOKE_FAILED, exit=$($start.ExitCode), log=$SmokeOut"
    exit 6
  }
  Write-Output "SMOKE_OK, log=$SmokeOut"
}

# Compute temp SHA
$TempSha = Get-SHA256 $Temp
Write-Output "TEMP_CREATED,$Temp,$TempSha"

# Preapply rename (preserve current target as preapply)
Move-Item -LiteralPath $Target -Destination $Preapply -Force
# Now move temp into place atomically
try {
  Move-Item -LiteralPath $Temp -Destination $Target -Force
} catch {
  # Attempt best-effort rollback
  Copy-Item -LiteralPath $Preapply -Destination $Target -Force
  Write-Output "SWAP_FAILED, rolled back to preapply: $Preapply"
  exit 7
}

# Finalize: compute after-sha and record ledger entry
$AfterSha = Get-SHA256 $Target
$actor = $env:USERNAME
$ci_job = $env:GITHUB_RUN_ID
$commit = ""
try { $commit = (& git rev-parse --short HEAD 2>$null).Trim() } catch {}
if (-not (Test-Path $Ledger)) {
  "timestamp,backup,orig_sha,after_sha,target,actor,ci_job,commit" | Out-File -LiteralPath $Ledger -Encoding UTF8
}
$entry = "{0},{1},{2},{3},{4},{5},{6},{7}" -f (Get-Date -Format o), $Backup, $OrigSha, $AfterSha, $Target, $actor, $ci_job, $commit
$entry | Out-File -LiteralPath $Ledger -Append -Encoding UTF8

# Emit JSON final summary (machine-parseable)
$summary = @{
  status = "OK"
  target = "$Target"
  backup = "$Backup"
  backup_sha256 = $OrigSha
  applied_sha256 = $AfterSha
  temp = "$Temp"
  temp_sha256 = $TempSha
  py_compile_log = "$PyCompileErr"
  smoke_log = if (Test-Path $SmokeOut) { "$SmokeOut" } else { $null }
  ledger = "$Ledger"
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $EditsDir "apply_summary.$ts.json") -Encoding UTF8

Write-Output "DONE, ledger=$Ledger, summary=$(Join-Path $EditsDir "apply_summary.$ts.json")"
Exit 0
