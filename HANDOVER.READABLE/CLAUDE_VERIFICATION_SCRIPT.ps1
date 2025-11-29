# CLAUDE_VERIFICATION_SCRIPT.ps1
# Verifies final bundle and manifests, prints closure entry and a JSON summary.
$finalZip = "F:\HO_HANDOVER\!!!_HO_HANDOVER_run-HO-20251123T155800-7F3C9D_20251123T171020_BUNDLE.zip"
$finalZipSidecar = $finalZip + ".sha256.txt"
$repoManifest = "C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\HANDOVER.RUN.manifest.json"
$readableManifest = "C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\HANDOVER.READABLE\HANDOVER.RUN.manifest.json"
$consol = "F:\HO_HANDOVER\CONSOLIDATION_SUMMARY_run-HO-20251123T155800-7F3C9D.json"
function Get-ShaLower($p){ if (Test-Path $p){ (Get-FileHash -Path $p -Algorithm SHA256).Hash.ToLower() } else { $null } }
function Read-SidecarLower($p){ if (Test-Path $p){ (Get-Content $p -Raw).Trim().ToLower() } else { $null } }
$report = [ordered]@{}
$report.final = @{ path=$finalZip; exists=Test-Path $finalZip; computed_sha = Get-ShaLower $finalZip; sidecar_sha = Read-SidecarLower $finalZipSidecar }
$report.repo_manifest = @{ path=$repoManifest; exists=Test-Path $repoManifest; computed_sha = Get-ShaLower $repoManifest; sidecar = if (Test-Path ($repoManifest + ".sha256.txt")) { (Get-Content ($repoManifest + ".sha256.txt") -Raw).Trim() } else { $null } }
$report.readable_manifest = @{ path=$readableManifest; exists=Test-Path $readableManifest; computed_sha = Get-ShaLower $readableManifest }
$report.consolidation = @{ path=$consol; exists=Test-Path $consol }
try { $jm = if (Test-Path $repoManifest){ Get-Content $repoManifest -Raw | ConvertFrom-Json } else { $null } } catch { $jm = $null }
$report.closure_entry = if ($jm -and $jm.handover -and $jm.handover.closure_entries) { $jm.handover.closure_entries[-1] } else { $null }
$report.timestamp = (Get-Date).ToString("o")
$report | ConvertTo-Json -Depth 12 | Set-Content -Path (Join-Path $env:TEMP "claude_verify_summary.json") -Encoding UTF8

