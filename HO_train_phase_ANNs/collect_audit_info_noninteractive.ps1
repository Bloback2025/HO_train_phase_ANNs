# collect_audit_info_noninteractive.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Fixed inputs
$repoRoot = "C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs"
$runIdFilter = $null   # set to a run folder name string to filter, or leave $null to collect all runs
$enableGit = $true
$scriptPatterns = @("train_phase2b2_HO.1B.py","train_phase2b2_HO.1Bi.py","train_phase2b2_HO.1A.py","train_phase2b2_HO.py","*2bANN2*.py")
$collectDirRoot = Join-Path $repoRoot "audit_collection"
$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$collectDir = Join-Path $collectDirRoot $ts
New-Item -ItemType Directory -Path $collectDir -Force | Out-Null

function NowIso { (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK") }
function FileSha256Upper([string]$path) { if (-not (Test-Path $path)) { return $null } ; (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToUpper() }

# Git info
Push-Location $repoRoot
$gitInfo = @{ present = $false; head_short = $null; head_full = $null; show_name_only = $null; status = $null }
if ($enableGit -and (Get-Command git -ErrorAction SilentlyContinue)) {
  $gitInfo.present = $true
  $gitInfo.head_short = (git rev-parse --short HEAD 2>$null).Trim()
  $gitInfo.head_full = (git rev-parse HEAD 2>$null).Trim()
  $gitInfo.show_name_only = (git show --name-only --pretty=format:"%h %ad %s" HEAD 2>$null) -join "`n"
  $gitInfo.status = (git status --porcelain 2>$null) -join "`n"
}
$gitInfo | ConvertTo-Json -Depth 5 | Out-File (Join-Path $collectDir "git_info.json") -Encoding utf8
Pop-Location

# Scripts discovery
$scriptsFound = @()
foreach ($pat in $scriptPatterns) {
  $found = Get-ChildItem -Path $repoRoot -Filter $pat -Recurse -ErrorAction SilentlyContinue -File
  foreach ($f in $found) {
    $sidecars = Get-ChildItem -Path $f.DirectoryName -Filter ($f.BaseName + ".sha256*") -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    $scriptsFound += [PSCustomObject]@{
      path = $f.FullName
      computed_sha256 = FileSha256Upper $f.FullName
      sidecar_paths = $sidecars
      sidecar_values = ($sidecars | ForEach-Object { (Get-Content $_ -Raw).Trim() })
      size = $f.Length
      last_write = $f.LastWriteTime.ToString("o")
    }
  }
}
$scriptsFound | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "scripts_summary.json") -Encoding utf8

# Locate ho_artifact_outputs run dirs
$runDirs = @()
$hoRoots = Get-ChildItem -Path $repoRoot -Filter "ho_artifact_outputs" -Recurse -Directory -ErrorAction SilentlyContinue
foreach ($h in $hoRoots) {
  if ($runIdFilter) {
    $candidate = Join-Path $h.FullName $runIdFilter
    if (Test-Path $candidate) { $runDirs += Get-Item $candidate }
  } else {
    $runDirs += Get-ChildItem -Path $h.FullName -Directory -ErrorAction SilentlyContinue
  }
}

# Collect artifacts per run
$runReports = @()
foreach ($rd in $runDirs) {
  $files = Get-ChildItem -Path $rd.FullName -File -Recurse -ErrorAction SilentlyContinue
  $artifacts = @()
  foreach ($f in $files) {
    $artifacts += [PSCustomObject]@{
      name = $f.Name
      fullpath = $f.FullName
      size = $f.Length
      last_write = $f.LastWriteTime.ToString("o")
      sha256_computed = FileSha256Upper $f.FullName
      sidecar_paths = (Get-ChildItem -Path $f.DirectoryName -Filter ($f.BaseName + ".*sha*") -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    }
  }
  $report = [PSCustomObject]@{ run_dir = $rd.FullName; artifacts = $artifacts }
  $runReports += $report
  $report | ConvertTo-Json -Depth 8 | Out-File (Join-Path $collectDir ("run_report_" + ($rd.Name) + ".json")) -Encoding utf8
}

# Retired folder summary
$retiredFolder = Join-Path $repoRoot "retired_ANN_files"
if (Test-Path $retiredFolder) {
  $retiredSummary = Get-ChildItem -Path $retiredFolder -File -ErrorAction SilentlyContinue | ForEach-Object {
    [PSCustomObject]@{
      name = $_.Name
      path = $_.FullName
      sha256 = FileSha256Upper $_.FullName
      size = $_.Length
      last_write = $_.LastWriteTime.ToString("o")
    }
  }
  $retiredSummary | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "retired_summary.json") -Encoding utf8
}

# Find archive_manifest.promotion.* files anywhere in repo
$archiveManifests = Get-ChildItem -Path $repoRoot -Filter "archive_manifest.promotion*.json" -Recurse -File -ErrorAction SilentlyContinue
$archiveList = $archiveManifests | ForEach-Object {
  [PSCustomObject]@{ path = $_.FullName; sha256 = FileSha256Upper $_.FullName; preview = (Get-Content $_.FullName -Raw) | Select-Object -First 1 }
}
$archiveList | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "archive_manifests_found.json") -Encoding utf8

# Capture named targets if present
$namedTargets = @("HANDOVER.RUN.manifest.json","HANDOVER.INDEX","ARCHIVAL_MANIFEST_2bANN2.txt")
$namedReport = @()
foreach ($n in $namedTargets) {
  $found = Get-ChildItem -Path $repoRoot -Filter $n -Recurse -File -ErrorAction SilentlyContinue
  foreach ($f in $found) {
    $namedReport += [PSCustomObject]@{
      name = $f.Name
      path = $f.FullName
      sha256 = FileSha256Upper $f.FullName
      first_lines = (Get-Content $f.FullName -TotalCount 40 -ErrorAction SilentlyContinue) -join "`n"
    }
  }
}
$namedReport | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "named_targets.json") -Encoding utf8

# Top-level summary
$summary = @{
  collected_at = NowIso
  repo_root = (Get-Item $repoRoot).FullName
  run_focus = $runIdFilter
  git_info_file = (Join-Path $collectDir "git_info.json")
  scripts_summary = (Join-Path $collectDir "scripts_summary.json")
  run_reports = ($runReports | ForEach-Object { $_.run_dir }) -join ";"
  retired_summary = if (Test-Path (Join-Path $collectDir "retired_summary.json")) { (Join-Path $collectDir "retired_summary.json") } else { $null }
  archive_manifests = (Join-Path $collectDir "archive_manifests_found.json")
  named_targets = (Join-Path $collectDir "named_targets.json")
  notes = "This collector reads files only and writes summaries under audit_collection. No commits or file moves are performed."
}
$summary | ConvertTo-Json -Depth 6 | Out-File (Join-Path $collectDir "collection_summary.json") -Encoding utf8

Write-Host "Collection complete. Output directory:" -ForegroundColor Green
Write-Host $collectDir
Write-Host "Primary summary file:" -ForegroundColor Cyan
Write-Host (Join-Path $collectDir "collection_summary.json")
