param(
  [Parameter(Mandatory=$true)][string]$ManifestPath,
  [Parameter(Mandatory=$true)][string]$ModelPath
)

try {
  $manifest = Get-Content -Raw -Path $ManifestPath | ConvertFrom-Json
} catch {
  Write-Error "FAILED: cannot read or parse manifest at $ManifestPath"
  exit 2
}

$required = @('commit_sha','branch','runlog_id','timestamp','rng_seed','env_image_tag','dependencies','model_sha256','test_verdicts')
$missing = $required | Where-Object { -not ($manifest.PSObject.Properties.Name -contains $_) -or [string]::IsNullOrWhiteSpace($manifest.$_) }

if ($missing) {
  Write-Error "FAILED: manifest missing required fields: $($missing -join ', ')"
  exit 3
}

if (-not (Test-Path -Path $ModelPath -PathType Leaf)) {
  Write-Error "FAILED: model file not found at $ModelPath"
  exit 4
}

try {
  $sha256 = Get-FileHash -Path $ModelPath -Algorithm SHA256 | Select-Object -ExpandProperty Hash
} catch {
  Write-Error "FAILED: error computing SHA256 for $ModelPath"
  exit 5
}

if ($sha256 -ne $manifest.model_sha256) {
  Write-Error "FAILED: SHA256 mismatch. Computed=$sha256 ; Manifest=$($manifest.model_sha256)"
  exit 6
}

$audit = [PSCustomObject]@{
  status = 'PASS'
  check = 'manifest_completeness_and_sha256'
  manifest = (Split-Path -Leaf $ManifestPath)
  model = (Split-Path -Leaf $ModelPath)
  computed_sha256 = $sha256
  manifest_sha256 = $manifest.model_sha256
  runlog_id = $manifest.runlog_id
  commit_sha = $manifest.commit_sha
  timestamp = $manifest.timestamp
  closure = "CLOSED:MANIFEST-SHA256:$($manifest.runlog_id)"
}
$audit | ConvertTo-Json -Compress
exit 0
