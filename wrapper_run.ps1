# wrapper_run.ps1 — skip-if-present; deterministic fallback; guarded finalize

# Resolve script base and runlogs dir
$ScriptBase = $PSScriptRoot
if (-not $ScriptBase) { $ScriptBase = (Resolve-Path ".").ProviderPath }
$RunlogsDir = Join-Path $ScriptBase "runlogs"
if (-not (Test-Path $RunlogsDir)) { New-Item -ItemType Directory -Path $RunlogsDir -Force | Out-Null }

function Write-Log {
  param([ValidateSet("INFO","WARN","ERROR","DEBUG")] [string]$level="INFO",[string]$step="",[string]$msg="",[hashtable]$extra=$null)
  try {
    $rec = @{ time=(Get-Date).ToString("o"); pid=$PID; level=$level; step=$step; msg=$msg }
    if ($extra) {
      $safe=@{}; foreach ($k in $extra.Keys) { $v=$extra[$k]; $safe[$k]= ($v -is [string] -or $v -is [int] -or $v -is [long] -or $v -is [double] -or $v -is [bool] -or $v -is [datetime]) ? $v : (($v | Out-String).Trim()) }
      $rec.extra = $safe
    }
    $json = $rec | ConvertTo-Json -Depth 6 -Compress
    [System.IO.File]::AppendAllText((Join-Path $RunlogsDir "extractor.log"), $json + "`n")
    Write-Output ("{0} {1} {2}" -f $rec.time, $rec.level, ($rec.msg -replace "`r?`n"," "))
  } catch {}
}

function Build-RunManifest {
  param([string]$RunId,[string[]]$Files,[string]$InputSource=$null,[int]$ExitCode=0,[string[]]$Errors=@())
  $manifest = @{ run_id=$RunId; start_time=(Get-Date).ToString("o"); pid=$PID; input_source=$InputSource; exit_code=$ExitCode; files=@(); errors=$Errors }
  foreach ($f in $Files) {
    if (-not (Test-Path $f)) { $manifest.files += @{ path=$f; present=$false }; continue }
    $fi = Get-Item $f; $hash = Get-FileHash -Path $f -Algorithm SHA256
    $type="TEXT"; $lines=$null; try { $lines=(Get-Content -LiteralPath $f -ErrorAction Stop).Count } catch {}
    $manifest.files += @{ path=$f; present=$true; size=$fi.Length; sha256=$hash.Hash; type=$type; lines=$lines }
  }
  $manifest.end_time = (Get-Date).ToString("o")
  $outfile = Join-Path $RunlogsDir ("RUN_{0}.json" -f $RunId)
  Set-Content -LiteralPath $outfile -Value ($manifest | ConvertTo-Json -Depth 6 -Compress) -Encoding utf8
  Write-Log -level INFO -step manifest -msg "MANIFEST_WRITTEN" -extra @{file=$outfile}
  return $outfile
}

function Create-Confirmed24Token {
  param([string]$FeatureFile = ".\runlogs\detected_feature_names.txt")
  $token = Join-Path $RunlogsDir "CONFIRMED_24.ok"
  if (-not (Test-Path $FeatureFile)) { return @{ ok=$false; reason="MISSING_FEATURE_FILE" } }
  try {
    $lines = Get-Content -LiteralPath $FeatureFile -ErrorAction Stop | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
    if ($lines.Count -eq 24) {
      Set-Content -LiteralPath $token -Value ("CONFIRMED {0} {1}" -f (Get-Date).ToString("o"), $PID) -Encoding utf8
      Write-Log -level INFO -step confirm -msg "CONFIRMED_24_CREATED"
      return @{ ok=$true; token=$token }
    } else {
      Write-Log -level WARN -step confirm -msg "NOT_CONFIRMED_24" -extra @{count=$lines.Count}
      return @{ ok=$false; reason="LINE_COUNT_MISMATCH"; count=$lines.Count }
    }
  } catch {
    Write-Log -level ERROR -step confirm -msg "CONFIRM_ERROR" -extra @{err=$_.Exception.Message}
    return @{ ok=$false; reason="ERROR"; err=$_.Exception.Message }
  }
}

function Acquire-ExtractorLock { param([int]$StaleSeconds=300)
  $lockfile = Join-Path $RunlogsDir "extractor.lock"
  try {
    $fs = [System.IO.File]::Open($lockfile,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
    $sw = New-Object System.IO.StreamWriter($fs); $sw.WriteLine((Get-Date).ToString("o")); $sw.WriteLine("pid=$PID"); $sw.Flush(); $sw.Dispose(); $fs.Dispose()
    Write-Log -level INFO -step lock -msg "LOCK_ACQUIRED" -extra @{file=$lockfile}; return $true
  } catch {
    if (Test-Path $lockfile) {
      try {
        $t = [DateTime]::Parse((Get-Content -LiteralPath $lockfile -ErrorAction Stop)[0])
        if ((Get-Date) - $t -gt (New-TimeSpan -Seconds $StaleSeconds)) { Remove-Item $lockfile -ErrorAction SilentlyContinue; Write-Log -level WARN -step lock -msg "LOCK_STALE_REMOVED"; return Acquire-ExtractorLock -StaleSeconds $StaleSeconds }
        else { Write-Log -level ERROR -step lock -msg "LOCK_PRESENT"; return $false }
      } catch { Write-Log -level ERROR -step lock -msg "LOCK_READ_ERROR" -extra @{err=$_.Exception.Message}; return $false }
    } else { Write-Log -level ERROR -step lock -msg "LOCK_CREATE_FAILED"; return $false }
  }
}
function Release-ExtractorLock { $lockfile = Join-Path $RunlogsDir "extractor.lock"; if (Test-Path $lockfile) { Remove-Item $lockfile -ErrorAction SilentlyContinue; Write-Log -level INFO -step lock -msg "LOCK_RELEASED" } }

# Main
$RunId = (Get-Date).ToString("yyyyMMdd-HHmmss")
Write-Log -level INFO -step main -msg "START" -extra @{run=$RunId}
if (-not (Acquire-ExtractorLock -StaleSeconds 300)) {
  Set-Content -LiteralPath (Join-Path $RunlogsDir "wrapper_summary.txt") -Value ((Get-Date).ToString("o") + " LOCK_PRESENT_EXIT") -Encoding utf8
  Write-Output ((Get-Date).ToString("o") + " LOCK_PRESENT_EXIT"); exit 6
}

$exitCode = 0; $errors = @()
$src = Join-Path $ScriptBase "train_phase2b2_HO_reinstated.py"
$mfPath = Join-Path $RunlogsDir "make_feature_extracted.txt"
$dfPath = Join-Path $RunlogsDir "detected_feature_names.txt"

try {
  # Skip if artifacts present and non-empty; otherwise perform deterministic extraction
  $mfReady = (Test-Path $mfPath) -and ((Get-Item $mfPath).Length -gt 0)
  $dfReady = (Test-Path $dfPath) -and ((Get-Item $dfPath).Length -gt 0)

  if (-not $mfReady -or -not $dfReady) {
    if (-not (Test-Path $src)) { Write-Log -level ERROR -step main -msg "MISSING_SOURCE" -extra @{file=$src}; $exitCode=1; throw "MISSING_SOURCE" }

    $lines = Get-Content -LiteralPath $src -Encoding utf8
    $startLine = 4
    $endLine = 2007
    if ($endLine -lt $startLine -or $endLine -gt $lines.Count) { Write-Log -level ERROR -step extract -msg "BOUNDARY_INVALID" -extra @{ start=$startLine; end=$endLine; total=$lines.Count }; $exitCode=3; throw "BOUNDARY_INVALID" }

    $bodyLines = $lines[($startLine-1)..($endLine-1)]
    Set-Content -LiteralPath $mfPath -Value $bodyLines -Encoding utf8

    $text = ($bodyLines -join "`n")
    $matches = [regex]::Matches($text, "['""]([A-Za-z0-9_\-\. ]{1,120})['""]")
    $candidates = @(); foreach ($m in $matches) { $candidates += $m.Groups[1].Value }
    $matches2 = [regex]::Matches($text, "\b(feature_list|features|cols|feature_names)\b\s*=\s*

\[([^\]

]+)\]

", "IgnoreCase")
    foreach ($m in $matches2) { $lit = $m.Groups[2].Value; $ms = [regex]::Matches($lit, "['""]([^'""]+)['""]"); foreach ($mm in $ms) { $candidates += $mm.Groups[1].Value } }
    $seen=@{}; $ordered=@(); foreach ($f in $candidates) { if (-not $seen.ContainsKey($f) -and $f -ne "") { $ordered += $f; $seen[$f]=1 } }
    Set-Content -LiteralPath $dfPath -Value $ordered -Encoding utf8
  }

  # Manifest + confirmation
  Build-RunManifest -RunId $RunId -Files @($mfPath,$dfPath) -InputSource $src -ExitCode 0 -Errors @() | Out-Null
  $confirmed = Create-Confirmed24Token -FeatureFile $dfPath
  if (-not $confirmed.ok) { $exitCode = 10 }

  Write-Log -level INFO -step main -msg "COMPLETE" -extra @{ run = $RunId; confirmed24 = $confirmed.ok }
  Write-Output ((Get-Date).ToString("o") + " COMPLETE")
}
catch {
  $errors += $_.Exception.Message
  $failureDir = Join-Path $RunlogsDir "failure_dumps"; if (-not (Test-Path $failureDir)) { New-Item -ItemType Directory -Path $failureDir | Out-Null }
  $dump = Join-Path $failureDir ("failure_{0}.txt" -f (Get-Date).ToString("yyyyMMdd-HHmmss"))
  Set-Content -LiteralPath $dump -Value ("RUNID: {0}`nPID:{1}`nTIME:{2}`nERRORS:`n{3}" -f $RunId,$PID,(Get-Date).ToString("o"),($errors -join "`n")) -Encoding utf8
  Write-Log -level ERROR -step main -msg "FAILED" -extra @{run=$RunId; errors=$errors}
  Write-Output ((Get-Date).ToString("o") + " FAILED")
  if ($exitCode -eq 0) { $exitCode = 127 }
}
finally {
  $allFiles=@()
  if ($null -ne $mfPath -and ($mfPath -ne "") -and (Test-Path $mfPath)) { $allFiles += $mfPath }
  if ($null -ne $dfPath -and ($dfPath -ne "") -and (Test-Path $dfPath)) { $allFiles += $dfPath }
  try { Build-RunManifest -RunId $RunId -Files $allFiles -InputSource $src -ExitCode $exitCode -Errors $errors | Out-Null } catch { Write-Log -level ERROR -step manifest -msg "FINAL_MANIFEST_WRITE_FAILED" -extra @{ err = $_.Exception.Message } }
  Release-ExtractorLock
  $hdr = (Get-Date).ToString("o") + " EXIT_CODE=" + $exitCode
  Set-Content -LiteralPath (Join-Path $RunlogsDir "wrapper_summary.txt") -Value $hdr -Encoding utf8
  Write-Output $hdr
  exit $exitCode
}
