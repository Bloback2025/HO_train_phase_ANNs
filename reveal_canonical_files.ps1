<#
reveal_canonical_files.ps1
Usage:
  .\reveal_canonical_files.ps1 -RepoUrl "https://github.com/Bloback2025/ANN_TEST_SUITE" -OutDir ".\tmp_repo" -KeepClone:$false

Parameters:
  -RepoUrl   : HTTPS git URL of the repo to inspect
  -OutDir    : Local folder to clone into (will be created)
  -KeepClone : If $true, do not delete the clone after inspection (default: $false)
#>

param(
  [Parameter(Mandatory=$true)][string]$RepoUrl,
  [string]$OutDir = ".\tmp_repo",
  [switch]$KeepClone
)

set -e

function LowerHexSha256OfFile($path) {
  $h = Get-FileHash -Algorithm SHA256 -Path $path
  return $h.Hash.ToLower()
}

# 1. Prepare outdir
$OutDir = Resolve-Path -Path $OutDir -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path -ErrorAction SilentlyContinue
if (-not $OutDir) { $OutDir = (Resolve-Path .).Path + "\tmp_repo_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
Write-Host "Cloning into: $OutDir"

# 2. Shallow clone (single branch, depth 1) to minimize network and disk
git clone --depth 1 $RepoUrl $OutDir
if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed"; exit 2 }

Push-Location $OutDir

try {
  # 3. Ensure we have a working tree and list candidate files tracked by git
  Write-Host "`n=== Tracked files matching canonical patterns ===`n"
  $patterns = @('*1B*.py','*1Bi*.py','*train*','*inference*','Handovers/*','HO_train_phase_ANNs/*')
  $matched = @()
  foreach ($p in $patterns) {
    $list = git ls-files $p 2>$null
    if ($list) {
      $list.Split("`n") | ForEach-Object { if ($_ -and -not ($matched -contains $_)) { $matched += $_ } }
    }
  }

  if ($matched.Count -eq 0) {
    Write-Host "No candidate files found for the given patterns. Listing all tracked files instead:`n"
    git ls-tree -r --name-only HEAD | Sort-Object
    Write-Host "`nYou can re-run with different patterns."
  } else {
    $table = @()
    foreach ($f in $matched | Sort-Object) {
      $full = Join-Path (Get-Location) $f
      if (Test-Path $full) {
        $sha = LowerHexSha256OfFile $full
        $lastCommit = git log -n 1 --pretty=format:"%h %ad %an %s" -- $f 2>$null
        $table += [PSCustomObject]@{
          Path = $f
          SHA256 = $sha
          "LastCommit" = $lastCommit
        }
      } else {
        $table += [PSCustomObject]@{
          Path = $f
          SHA256 = "<missing>"
          "LastCommit" = "<no-commit>"
        }
      }
    }
    $table | Format-Table -AutoSize
  }

  # 4. Show recent commits that touched any of these files
  Write-Host "`n=== Recent commits touching candidate files (last 20) ===`n"
  if ($matched.Count -gt 0) {
    git log -n 20 --pretty=format:"%h %ad %an %s" --date=iso -- $matched
  } else {
    git log -n 20 --pretty=format:"%h %ad %an %s" --date=iso
  }

  # 5. Show Handovers folder contents if present
  Write-Host "`n=== Handovers folder (if present) ===`n"
  if (Test-Path "Handovers") {
    git ls-tree -r --name-only HEAD Handovers | Sort-Object | ForEach-Object {
      $p = $_
      $full = Join-Path (Get-Location) $p
      $sha = if (Test-Path $full) { LowerHexSha256OfFile $full } else { "<missing>" }
      Write-Host "$p`t$sha"
    }
  } else {
    Write-Host "No Handovers folder found in this commit."
  }

  # 6. Show submodule status (if any)
  Write-Host "`n=== Submodule status ===`n"
  git submodule status --recursive 2>$null
  if ($LASTEXITCODE -ne 0) { Write-Host "No submodules or git not reporting submodule status." }

  # 7. Show HEAD commit and branch
  Write-Host "`n=== Repo HEAD and branch ===`n"
  $head = git rev-parse --verify HEAD
  $branch = git rev-parse --abbrev-ref HEAD
  Write-Host "Branch: $branch"
  Write-Host "HEAD: $head"

  # 8. Optional: compute script SHA for a specific file if requested interactively
  Write-Host "`nIf you want a single file SHA, run:`n  Get-FileHash -Algorithm SHA256 <path> | Select-Object -ExpandProperty Hash`n"
}
finally {
  Pop-Location
  if (-not $KeepClone) {
    Write-Host "`nCleaning up: removing clone at $OutDir"
    Remove-Item -Recurse -Force $OutDir
  } else {
    Write-Host "`nKeeping clone at $OutDir (KeepClone set)."
  }
}
