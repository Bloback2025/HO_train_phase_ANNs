# collect_audit_info.ps1
# Interactive collector for audit artifacts and provenance info
# Writes output to: <repoRoot>\audit_collection\<ISO_TS>\

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function NowIso { (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK") }

Write-Host "=== Audit collector ===" -ForegroundColor Cyan

# Ask user for repo root (full path)
$repoRoot = Read-Host "Enter repository root full path (e.g. C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080)"
if (-not (Test-Path $repoRoot)) { Write-Error "Path not found: $repoRoot"; exit 2 }

# Optional run_id or run outdir name to focus on
$runIdInput = Read-Host "Optional: enter run_id or run outdir name to focus on (leave empty to collect all runs)"
if ($runIdInput -eq "") { $runIdInput = $null }

# Collector output dir
$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$collectDir = Join-Path -Path $repoRoot -ChildPath ("audit_collection\" + $ts)
New-Item -ItemType Directory -Path $collectDir -Force | Out-Null

# Helper: compute SHA uppercase
function FileSha256Upper([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  return (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToUpper()
}

# 1) Git context (if repo)
Push-Location $repoRoot
$gitInfo = @{
  present = $false
  head_short = $null
  head_full = $null
  show_name_only = $null
  status = $null
}
try {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitInfo.present = $true
    $gitInfo.head_short = (git rev-parse --short HEAD 2>$null).Trim()
    $gitInfo.head_full = (git rev-parse HEAD 2>$null).Trim()
    $gitInfo.show_name_only = (git show --name-only --pretty=format:"%h %ad %s" HEAD 2>$null) -join "`n"
    $gitInfo.status = (git status --porcelain 2>$null) -join "`n"
    # Save raw outputs
    $gitInfo | ConvertTo-Json -Depth 5 | Out-File (Join-Path $collectDir "git_info.json") -Encoding utf8
  } else {
    $gitInfo.present = $false
    $gitInfo.status = "git not found in PATH"
    $gitInfo | ConvertTo-Json -Depth 5 | Out-File (Join-Path $collectDir "git_info.json") -Encoding utf8
  }
} catch {
  $gitInfo.error = $_.ToString()
  $gitInfo | ConvertTo-Json -Depth 5 | Out-File (Join-Path $collectDir "git_info.json") -Encoding utf8
}
Pop-Location

# 2) Find canonical script files matching known patterns
$scriptPatterns = @("train_phase2b2_HO.1B.py","train_phase2b2_HO.1Bi.py","train_phase2b2_HO.1A.py","train_phase2b2_HO.py","*2bANN2*.py")
$scriptsFound = @()
foreach ($pat in $scriptPatterns) {
  $found = Get-ChildItem -Path $repoRoot -Filter $pat -Recurse -ErrorAction SilentlyContinue -File
  foreach ($f in $found) {
    $scriptsFound += [PSCustomObject]@{
      path = $f.FullName
      sha256_sidecar_paths = @( (Get-ChildItem -Path $f.DirectoryName -Filter ($f.BaseName + ".sha256*") -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }) )
      computed_sha256 = FileSha256Upper $f.FullName
      sidecar_sha_values = @( (Get-ChildItem -Path $f.DirectoryName -Filter ($f.BaseName + ".sha256*") -File -ErrorAction SilentlyContinue | ForEach-Object { (Get-Content $_.FullName -Raw).Trim() }) )
      size = $f.Length
      mtime = $f.LastWriteTime.ToString("o")
    }
  }
}
# Save scripts summary
$scriptsFound | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "scripts_summary.json") -Encoding utf8

# 3) Locate run outdirs under ho_artifact_outputs or provided run id
$hoRootCandidates = @(
  Join-Path $repoRoot "ho_artifact_outputs",
  Join-Path $repoRoot "ho_artifact_outputs".ToUpper(),
  Join-Path $repoRoot "ho_artifact_outputs".ToLower()
) | Where-Object { Test-Path $_ } | Select-Object -Unique

$runDirs = @()
if ($hoRootCandidates) {
  foreach ($hoRoot in $hoRootCandidates) {
    if ($runIdInput) {
      $candidate = Join-Path $hoRoot $runIdInput
      if (Test-Path $candidate) { $runDirs += Get-Item $candidate }
    } else {
      $runDirs += Get-ChildItem -Path $hoRoot -Directory -ErrorAction SilentlyContinue
    }
  }
}

# If none found, also search for ho_artifact_outputs anywhere under repoRoot
if (-not $runDirs) {
  $foundHo = Get-ChildItem -Path $repoRoot -Filter "ho_artifact_outputs" -Recurse -Directory -ErrorAction SilentlyContinue
  foreach ($h in $foundHo) {
    if ($runIdInput) {
      $candidate = Join-Path $h.FullName $runIdInput
      if (Test-Path $candidate) { $runDirs += Get-Item $candidate }
    } else {
      $runDirs += Get-ChildItem -Path $h.FullName -Directory -ErrorAction SilentlyContinue
    }
  }
}

# 4) For each run dir collect artifact files, compute SHAs, and capture sidecars
$runReports = @()
foreach ($rd in $runDirs) {
  $files = Get-ChildItem -Path $rd.FullName -File -Recurse -ErrorAction SilentlyContinue
  $artifacts = @()
  foreach ($f in $files) {
    $sha = $null
    try { $sha = FileSha256Upper $f.FullName } catch { $sha = $null }
    $sidecars = Get-ChildItem -Path $f.DirectoryName -Filter ($f.BaseName + ".*sha*") -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    $artifacts += [PSCustomObject]@{
      name = $f.Name
      relpath = $f.FullName.Replace((Get-Location).Path + [IO.Path]::DirectorySeparatorChar, "")
      size = $f.Length
      last_write = $f.LastWriteTime.ToString("o")
      sha256_computed = $sha
      sidecar_paths = $sidecars
    }
  }
  $runReports += [PSCustomObject]@{
    run_dir = $rd.FullName
    artifacts = $artifacts
  }
  # save per-run artifact list
  $runReports[-1] | ConvertTo-Json -Depth 8 | Out-File (Join-Path $collectDir ("run_report_" + ($rd.Name) + ".json")) -Encoding utf8
}

# 5) Search for archival manifests and retired_ANN_files
$retiredFolder = Join-Path $repoRoot "retired_ANN_files"
$retiredFound = @()
if (Test-Path $retiredFolder) {
  $retiredFound = Get-ChildItem -Path $retiredFolder -File -ErrorAction SilentlyContinue
  $retiredSummary = @()
  foreach ($f in $retiredFound) {
    $retiredSummary += [PSCustomObject]@{
      name = $f.Name
      path = $f.FullName
      sha256 = FileSha256Upper $f.FullName
      last_write = $f.LastWriteTime.ToString("o")
      size = $f.Length
    }
  }
  $retiredSummary | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "retired_summary.json") -Encoding utf8
}

# 6) Capture any archive_manifest.promotion.* files anywhere in repo
$archiveManifests = Get-ChildItem -Path $repoRoot -Filter "archive_manifest.promotion*.json" -Recurse -File -ErrorAction SilentlyContinue
$archiveList = @()
foreach ($am in $archiveManifests) {
  $archiveList += [PSCustomObject]@{
    path = $am.FullName
    sha256 = FileSha256Upper $am.FullName
    content_preview = (Get-Content $am.FullName -Raw) | Select-Object -First 1
  }
}
$archiveList | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "archive_manifests_found.json") -Encoding utf8

# 7) Capture specific named files if present (HANDOVER.RUN.manifest.json, HANDOVER.INDEX, ARCHIVAL_MANIFEST_2bANN2.txt)
$namedTargets = @("HANDOVER.RUN.manifest.json","HANDOVER.INDEX","ARCHIVAL_MANIFEST_2bANN2.txt","ARCHIVAL_MANIFEST_2bANN2.TXT")
$namedReport = @()
foreach ($n in $namedTargets) {
  $found = Get-ChildItem -Path $repoRoot -Filter $n -Recurse -File -ErrorAction SilentlyContinue
  foreach ($f in $found) {
    $namedReport += [PSCustomObject]@{
      name = $f.Name
      path = $f.FullName
      sha256 = FileSha256Upper $f.FullName
      first_lines = (Get-Content $f.FullName -TotalCount 20 -ErrorAction SilentlyContinue) -join "`n"
    }
  }
}
$namedReport | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "named_targets.json") -Encoding utf8

# 8) Summarise findings and write top-level report
$summary = @{
  collected_at = NowIso
  repo_root = (Get-Item $repoRoot).FullName
  run_focus = $runIdInput
  git_info_file = (Join-Path $collectDir "git_info.json")
  scripts_summary = (Join-Path $collectDir "scripts_summary.json")
  runs_collected = ($runReports | ForEach-Object { $_.run_dir }) -join ";"
  retired_summary = if (Test-Path (Join-Path $collectDir "retired_summary.json")) { (Join-Path $collectDir "retired_summary.json") } else { $null }
  archive_manifests = (Join-Path $collectDir "archive_manifests_found.json")
  named_targets = (Join-Path $collectDir "named_targets.json")
  notes = "Do not commit or move files; this script only collects and writes summary files under audit_collection. Provide outputs to the reviewer for verification before committing any changes."
}
$summary | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "collection_summary.json") -Encoding utf8

Write-Host "Collection complete. Output directory:" -ForegroundColor Green
Write-Host $collectDir
Write-Host "Files produced:" -ForegroundColor Cyan
Get-ChildItem -Path $collectDir -File | Select-Object Name,Length | Format-Table -AutoSize
