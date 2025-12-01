# HANDOVER.READABLE\FIX_AND_RUN_SMOKE.ps1
# Purpose: deterministically run smoke described by smoke\deterministic_inference_smoke.json
$Repo = "C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs"
$JsonPath = Join-Path $Repo "smoke\deterministic_inference_smoke.json"
$WorkLogDir = "F:\CLAUDE_MESSAGES"
$SmokeLog = Join-Path $WorkLogDir "smoke_run_output.log"
$VerifySummary = Join-Path $WorkLogDir "auto_verify_summary.json"
$SmokeOut = Join-Path $Repo "deterministic_inference_outputs"
$Manifest = "HANDOVER.RUN.manifest.json"
$Seed = 42

function Write-Log($m){ $t=(Get-Date -Format o); "$t $m" | Tee-Object -FilePath $SmokeLog -Append; Write-Output $m }
function FileSha($p){ if (Test-Path $p){ (Get-FileHash -Path $p -Algorithm SHA256).Hash.ToLower() } else { $null } }

New-Item -Path (Split-Path $SmokeLog) -ItemType Directory -Force | Out-Null
New-Item -ItemType Directory -Path $SmokeOut -Force | Out-Null

if (-not (Test-Path $JsonPath)) { Write-Log "ERROR JSON_SMOKE_NOT_FOUND -> $JsonPath"; exit 2 }

try {
  $raw = Get-Content $JsonPath -Raw
  $j = $raw | ConvertFrom-Json
} catch {
  Write-Log "ERROR JSON_PARSE -> $($_.Exception.Message)"
  exit 3
}

# discover explicit runnable info
$runnable = $null
foreach ($k in $j.PSObject.Properties.Name) {
  $lname = $k.ToLower()
  if ($lname -in @("entrypoint","script","command","cmd","runnable","python","module","run")) {
    $val = $j.$k
    if ($val -is [string] -and $val.Trim().Length -gt 0) { $runnable = $val.Trim(); break }
  }
}

# fallback discovery: look for common scripts in repo
$defaultCandidates = @(
  Join-Path $Repo "scripts\run_smoke.py",
  Join-Path $Repo "scripts\run.py",
  Join-Path $Repo "scripts\smoke_run.py",
  Join-Path $Repo "run_smoke.py",
  Join-Path $Repo "run.py",
  Join-Path $Repo "smoke_run.py"
)
foreach ($c in $defaultCandidates) { if (-not $runnable -and (Test-Path $c)) { $runnable = $c } }

if (-not $runnable) {
  $outNote = Join-Path $Repo "HANDOVER.READABLE\SMOKE_MANUAL_REMEDIATION.txt"
  $note = "No runnable discovered. Inspect smoke\deterministic_inference_smoke.json and run the intended script manually with: python -u <script.py> --manifest $Manifest --out deterministic_inference_outputs --seed $Seed"
  $note | Set-Content -Path $outNote -Encoding UTF8
  Write-Log "NO_RUNNABLE_FOUND -> wrote $outNote"
  exit 5
}

if ($runnable -match "\.json(\b|$)") {
  $outNote = Join-Path $Repo "HANDOVER.READABLE\SMOKE_MANUAL_REMEDIATION.txt"
  "Detected runnable field points to JSON. Inspect $JsonPath and update entrypoint." | Set-Content -Path $outNote -Encoding UTF8
  Write-Log "RUNNABLE_IS_JSON -> wrote $outNote"
  exit 6
}

# normalize candidate path and build command
$execCmd = $null
$ext = [IO.Path]::GetExtension($runnable).ToLower()
if ($ext -eq ".py" -or $runnable -match "(?i)\bpython\b") {
  $scriptPath = if (Test-Path $runnable) { $runnable } elseif (Test-Path (Join-Path $Repo $runnable)) { Join-Path $Repo $runnable } else { $runnable }
  $execCmd = "python -u `"$scriptPath`" --manifest $Manifest --out `"$SmokeOut`" --seed $Seed"
} elseif ($ext -eq ".ps1") {
  $scriptPath = if (Test-Path $runnable) { $runnable } else { Join-Path $Repo $runnable }
  $execCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" --manifest $Manifest --out `"$SmokeOut`" --seed $Seed"
} else {
  $execCmd = $runnable + " --manifest $Manifest --out `"$SmokeOut`" --seed $Seed"
}

Write-Log "EXEC_PLAN -> $execCmd"
Write-Log "START_EXECUTION"
try {
  & cmd /c $execCmd 2>&1 | Tee-Object -FilePath $SmokeLog -Append
  $exit = $LASTEXITCODE
  Write-Log ("EXEC_EXITCODE -> " + $exit)
} catch {
  Write-Log "EXEC_EXCEPTION -> $($_.Exception.Message)"
  $exit = 1
}

$preds = Join-Path $SmokeOut "preds.json"
$predsSha = FileSha $preds
$realPreds = $false
if (Test-Path $preds) {
  $txt = Get-Content $preds -Raw
  if ($txt -and ($txt -notmatch "AUTO_GENERATED")) { $realPreds = $true }
}

$summary = @{
  timestamp = (Get-Date).ToString("o")
  runnable = $runnable
  exec_cmd = $execCmd
  exit_code = $exit
  preds_path = $preds
  preds_sha = $predsSha
  realPreds = $realPreds
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $VerifySummary -Encoding UTF8
Write-Log "WROTE_VERIFY_SUMMARY -> $VerifySummary"

if ($realPreds) { Write-Log "REAL_PREDs_DETECTED -> $preds ; SHA -> $predsSha"; exit 0 }
Write-Log "NO_REAL_PREDs -> exit $exit"
exit 7
