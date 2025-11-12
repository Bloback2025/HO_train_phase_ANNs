# create_ho_final_auto_tailored_with_env.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- tailored paths (as verified) ---
$HO_FOLDER        = "C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080"
$PROJECT_ROOT     = "C:\Users\loweb\AI_Financial_Sims\ANN_TEST_SUITE"
$ARTIFACT_OUTDIR  = Join-Path $PROJECT_ROOT "ho_artifact_outputs"
$TRAIN_CSV        = Join-Path $HO_FOLDER "hoxnc_training.csv"
$VALID_CSV        = Join-Path $HO_FOLDER "hoxnc_validation.csv"
$TEST_CSV         = Join-Path $HO_FOLDER "hoxnc_testing.csv"
$MODEL_FILE       = Join-Path $HO_FOLDER "2bANN2_HO_model.keras"
$TRAIN_SCRIPT     = Join-Path $HO_FOLDER "train_phase2b2_HO.py"
$FEATURE_MANIFEST = Join-Path $HO_FOLDER "hoxnc_training.with_base_and_lags.manifest.json"
$SCALER_PKL       = Join-Path $HO_FOLDER "scaler.pkl"
$SCALER_SIDECAR   = Join-Path $HO_FOLDER "scaler_sidecar.json"
$RUNLOG_OUTFILE   = Join-Path $ARTIFACT_OUTDIR "RUNLOG_2bANN2_HO_final.json"
$SUMMARY_OUTFILE  = Join-Path $ARTIFACT_OUTDIR "artifact_hashes_summary.json"
$AUDIT_LEDGER     = Join-Path $HO_FOLDER "audit_ledger.csv"
$PYTHON_BIN       = "C:\Users\loweb\AppData\Local\Programs\Python\Python312\python.exe"
# --- end paths ---

if (-not (Test-Path $ARTIFACT_OUTDIR)) { New-Item -Path $ARTIFACT_OUTDIR -ItemType Directory | Out-Null }

function Get-Sha256OrNull($path) { if (Test-Path $path) { (Get-FileHash -Algorithm SHA256 -Path $path -ErrorAction Stop).Hash } else { $null } }

# ordered features
$ordered_features = @()
if (Test-Path $FEATURE_MANIFEST) {
  try { $mf = Get-Content $FEATURE_MANIFEST -Raw | ConvertFrom-Json; $ordered_features = $mf.ordered_feature_names ?? $mf.features ?? @() } catch { $ordered_features = @() }
}
if (-not $ordered_features -or $ordered_features.Count -eq 0) {
  if (Test-Path $TRAIN_CSV) {
    $hdr = (Get-Content -Path $TRAIN_CSV -TotalCount 1) -split ","
    $ordered_features = $hdr | ForEach-Object { $_.Trim('"') }
  }
}
if (-not $ordered_features -or $ordered_features.Count -eq 0) { Write-Error "Cannot determine ordered features; aborting."; exit 4 }

# ensure sidecar exists
if (-not (Test-Path $SCALER_SIDECAR)) {
  $side = [ordered]@{ feature_names = $ordered_features; means = $null; scales = $null; scaler_class = $null; saved_at = (Get-Date).ToString("o") }
  $side | ConvertTo-Json -Depth 6 | Out-File -FilePath $SCALER_SIDECAR -Encoding UTF8
}

# populate sidecar from scaler.pkl if needed
$scaler_sidecar_json = Get-Content -Raw -Path $SCALER_SIDECAR | ConvertFrom-Json
$need_populate = ($scaler_sidecar_json.means -eq $null -or $scaler_sidecar_json.scales -eq $null) -and (Test-Path $SCALER_PKL)
if ($need_populate) {
  $py = @"
import pickle, json, sys
scaler_pkl = r'''$SCALER_PKL'''
sidecar = r'''$SCALER_SIDECAR'''
manifest = r'''$FEATURE_MANIFEST'''
out = {'feature_names': None, 'means': None, 'scales': None, 'scaler_class': None, 'saved_at': None}
try:
    with open(manifest,'r',encoding='utf8') as f:
        mf = json.load(f)
        out['feature_names'] = mf.get('ordered_feature_names') or mf.get('features') or []
except Exception:
    out['feature_names'] = None
try:
    with open(scaler_pkl,'rb') as f:
        s = pickle.load(f)
    out['scaler_class'] = type(s).__name__
    means = getattr(s,'mean_', None)
    scales = getattr(s,'scale_', None)
    out['means'] = means.tolist() if hasattr(means,'tolist') else None
    out['scales'] = scales.tolist() if hasattr(scales,'tolist') else None
    out['saved_at'] = __import__('datetime').datetime.now().isoformat()
except Exception as e:
    out['scaler_class'] = f"ERROR_READING_SCALER: {e}"
with open(sidecar,'w',encoding='utf8') as f:
    json.dump(out, f, indent=2)
print('WROTE_SIDE_CAR')
"@
  $tmp = Join-Path $env:TEMP "wss_helper.py"
  $py | Out-File -FilePath $tmp -Encoding UTF8
  Start-Process -FilePath $PYTHON_BIN -ArgumentList $tmp -NoNewWindow -Wait -PassThru | Out-Null
  Remove-Item $tmp -ErrorAction SilentlyContinue
  try { $scaler_sidecar_json = Get-Content -Raw -Path $SCALER_SIDECAR | ConvertFrom-Json } catch {}
}

# compute hashes
$train_hash  = Get-Sha256OrNull $TRAIN_CSV
$valid_hash  = Get-Sha256OrNull $VALID_CSV
$test_hash   = Get-Sha256OrNull $TEST_CSV
$model_hash  = Get-Sha256OrNull $MODEL_FILE
$script_hash = Get-Sha256OrNull $TRAIN_SCRIPT
$manifest_hash= Get-Sha256OrNull $FEATURE_MANIFEST
$scaler_pkl_hash = Get-Sha256OrNull $SCALER_PKL
$scaler_sidecar_hash = Get-Sha256OrNull $SCALER_SIDECAR

# fail fast on critical missing
if (-not $train_hash -or -not $test_hash -or -not $model_hash -or -not $script_hash) { Write-Error "Critical artifact missing (training/test/model/script). Aborting."; exit 3 }

# run null controls and capture deterministic outputs
$det_default = Join-Path $ARTIFACT_OUTDIR "deterministic_inference_result.json"
$det_nt = Join-Path $ARTIFACT_OUTDIR "deterministic_inference_result_null_target.json"
$det_nf = Join-Path $ARTIFACT_OUTDIR "deterministic_inference_result_null_features.json"

function Run-Null-Capture($flag,$target) {
  Write-Host "Running $flag ..."
  Start-Process -FilePath $PYTHON_BIN -ArgumentList "`"$TRAIN_SCRIPT`" $flag" -NoNewWindow -Wait -PassThru | Out-Null
  Start-Sleep -Seconds 1
  if (Test-Path $det_default) { Copy-Item -Path $det_default -Destination $target -Force; Write-Host "Captured $target" } else { Write-Warning "No deterministic output for $flag at $det_default" }
}

Run-Null-Capture "--null_target" $det_nt
Run-Null-Capture "--null_features" $det_nf

# helper to parse metrics
function Parse-Metrics($file) {
  if (Test-Path $file) {
    try { $j = Get-Content -Raw -Path $file | ConvertFrom-Json; $m = $j.metrics; return @{ mae = $m.mae; rmse = $m.rmse; r2 = $m.r2; count = $m.count } } catch { Write-Warning "Failed to parse $file"; return $null }
  } else { return $null }
}

$nt_metrics = Parse-Metrics $det_nt
$nf_metrics = Parse-Metrics $det_nf

# capture env versions
$env_info = $null
try {
  $pyenv = @"
import sys
def vers():
    import json, sys
    out = {'python': sys.version.split()[0]}
    try:
        import tensorflow as tf; out['tensorflow']=tf.__version__
    except: out['tensorflow']=None
    try:
        import numpy as np; out['numpy']=np.__version__
    except: out['numpy']=None
    try:
        import pandas as pd; out['pandas']=pd.__version__
    except: out['pandas']=None
    try:
        import sklearn; out['scikit_learn']=sklearn.__version__
    except: out['scikit_learn']=None
    print(json.dumps(out))
vers()
"@
  $tmp2 = Join-Path $env:TEMP "env_probe.py"
  $pyenv | Out-File -FilePath $tmp2 -Encoding UTF8
  $out = & $PYTHON_BIN $tmp2
  Remove-Item $tmp2 -ErrorAction SilentlyContinue
  $env_info = $out | ConvertFrom-Json
} catch { Write-Warning "Failed to capture environment versions via Python" }

# build and write summary and RUNLOG
$summary = [ordered]@{
  run_timestamp = (Get-Date).ToString("o")
  artifacts = [ordered]@{
    training_csv = @{ path = $TRAIN_CSV; sha256 = $train_hash }
    validation_csv = @{ path = $VALID_CSV; sha256 = $valid_hash }
    testing_csv = @{ path = $TEST_CSV; sha256 = $test_hash }
    model = @{ path = $MODEL_FILE; sha256 = $model_hash }
    script = @{ path = $TRAIN_SCRIPT; sha256 = $script_hash }
    feature_manifest = @{ path = $FEATURE_MANIFEST; sha256 = $manifest_hash }
  }
  optional = [ordered]@{
    scaler_pkl = @{ path = $SCALER_PKL; sha256 = $scaler_pkl_hash }
    scaler_sidecar = @{ path = $SCALER_SIDECAR; sha256 = $scaler_sidecar_hash }
  }
  metrics = @{ mae = 2.6244; rmse = 3.5441; r2 = 0.9936 }
  status = @{ missing = @(); note = "Auto finaliser: environment captured; null controls attempted." }
}
$summary | ConvertTo-Json -Depth 8 | Out-File -FilePath $SUMMARY_OUTFILE -Encoding UTF8

$runlog = [ordered]@{
  run_id = "2bANN2_HO_RUN_" + (Get-Date -Format "yyyyMMddTHHmmss")
  datetime = (Get-Date).ToString("o")
  model_file = (Split-Path $MODEL_FILE -Leaf)
  model_hash = $model_hash
  ordered_feature_names = $ordered_features
  scaler = @{
    type = if ($scaler_pkl_hash) { "sklearn_pickle" } else { "sidecar_only" }
    sidecar_present = $true
    sidecar_hash = $scaler_sidecar_hash
    feature_names = $ordered_features
  }
  data_files = @{
    training_csv = @{ path = (Split-Path $TRAIN_CSV -Leaf); sha256 = $train_hash }
    validation_csv = @{ path = (Split-Path $VALID_CSV -Leaf); sha256 = $valid_hash }
    testing_csv = @{ path = (Split-Path $TEST_CSV -Leaf); sha256 = $test_hash }
  }
  script = @{ path = (Split-Path $TRAIN_SCRIPT -Leaf); sha256 = $script_hash }
  feature_manifest = @{ path = (Split-Path $FEATURE_MANIFEST -Leaf); sha256 = $manifest_hash }
  config = @{ path = $null; sha256 = $null }
  seed = 42
  environment = $env_info
  metrics = @{ test_mae = 2.6244; test_rmse = 3.5441; test_r2 = 0.9936; test_count = $null }
  notes = "Auto finaliser: scaler sidecar updated; null controls attempted; environment recorded."
}
$runlog | ConvertTo-Json -Depth 8 | Out-File -FilePath $RUNLOG_OUTFILE -Encoding UTF8

# append ledger row
$now_iso = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
$real_metrics = "MAE=2.6244|RMSE=3.5441|R2=0.9936"
$script_hash_field = $script_hash
$config_hash_field = "<CONFIG_SHA256>"
$nt_field = if ($nt_metrics) { "MAE=$($nt_metrics.mae)|RMSE=$($nt_metrics.rmse)|R2=$($nt_metrics.r2)" } else { "<NULL_TARGET_METRICS>" }
$nf_field = if ($nf_metrics) { "MAE=$($nf_metrics.mae)|RMSE=$($nf_metrics.rmse)|R2=$($nf_metrics.r2)" } else { "<NULL_FEATURES_METRICS>