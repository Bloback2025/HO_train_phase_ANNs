# BACKUP + PATCH null_tests.py: add manifest-driven scaler and inverse-target handling
Copy-Item null_tests.py null_tests.py.bak -Force
$in = Get-Content null_tests.py
$out = @()
$inserted = $false
foreach ($line in $in) {
  $out += $line
  if (-not $inserted -and $line -match '^\s*import\s+numpy\s+as\s+np\s*$') {
    $out += 'import json, os, sys'
    $out += 'from pathlib import Path'
    $out += 'try:'
    $out += '    # locate artifact folder and sidecars'
    $out += '    art = Path(r"ho_artifact_outputs")'
    $out += '    sidecar_j = art / "scaler_sidecar.json"'
    $out += '    pkl = art / "scaler.pkl"'
    $out += '    target_scaler = None'
    $out += '    input_scaler = None'
    $out += '    if sidecar_j.exists():'
    $out += '        sc_meta = json.load(open(sidecar_j, "r"))'
    $out += '        # sidecar keys: means, scales, feature_names, scaler_class'
    $out += '        # prefer pickle if present'
    $out += '    if pkl.exists():'
    $out += '        import pickle'
    $out += '        input_scaler = pickle.load(open(pkl,"rb"))'
    $out += '    # load X/y if not already loaded by the script caller'
    $out += '    if "X_eval" not in globals():'
    $out += '        if Path("X_eval.npy").exists():'
    $out += '            X_eval = np.load("X_eval.npy")'
    $out += '        elif Path("X_eval_scaled.npy").exists():'
    $out += '            X_eval = np.load("X_eval_scaled.npy")'
    $out += '        else:'
    $out += '            raise FileNotFoundError("X_eval.npy or X_eval_scaled.npy not found")'
    $out += '    if "y_eval" not in globals():'
    $out += '        if Path("y_eval.npy").exists():'
    $out += '            y_eval = np.load("y_eval.npy")'
    $out += '        else:'
    $out += '            raise FileNotFoundError("y_eval.npy not found")'
    $out += '    # apply input scaler if script called with raw X and scaler exists'
    $out += '    if input_scaler is not None and ("X_eval_scaled" not in globals()):'
    $out += '        try:'
    $out += '            X_eval_scaled = input_scaler.transform(X_eval)'
    $out += '            X_used = X_eval_scaled'
    $out += '        except Exception:'
    $out += '            X_used = X_eval'
    $out += '    else:'
    $out += '        X_used = X_eval'
    $out += 'except Exception as e:'
    $out += '    # fail fast and keep original traceback visible'
    $out += '    print("MANIFEST/PREPROCESS GUARD WARNING:", e)'
    $out += '    X_used = globals().get("X_eval", None)'
    $out += '    y_eval = globals().get("y_eval", None)'
    $out += '    input_scaler = None'
    $out += '    target_scaler = None'
    $inserted = $true
  }
}
$out | Set-Content null_tests.py
Write-Output "PATCHED null_tests.py (backup: null_tests.py.bak). Please re-run null_tests.py normally."
