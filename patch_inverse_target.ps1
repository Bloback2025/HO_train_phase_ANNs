# BACKUP + PATCH null_tests.py: add manifest-driven inverse-target handling
Copy-Item null_tests.py null_tests.py.bak -Force
$in = Get-Content null_tests.py
$out = @()
$inserted = $false
foreach ($line in $in) {
  $out += $line
  if (-not $inserted -and $line -match '^\s*#\s*null-input R2:' -or -not $inserted -and $line -match '^\s*#\s*null-target') {
    # no-op: avoid accidental match; continue scanning
  }
  # Insert inverse-target block just before the final scoring section where y_pred is used.
  if (-not $inserted -and $line -match 'y_pred\s*=\s*model\.predict') {
    $out += '    # If a target scaler exists in artifacts, inverse-transform predictions before scoring'
    $out += '    try:'
    $out += '        import pickle'
    $out += '        from pathlib import Path'
    $out += '        artp = Path(r"ho_artifact_outputs")'
    $out += '        tsc_p = artp / "target_scaler.pkl"'
    $out += '        # fallback to generic scaler.pkl only if it clearly represents a target scaler; do not assume by default'
    $out += '        if not tsc_p.exists():'
    $out += '            alt = artp / "scaler.pkl"'
    $out += '            # only treat alt as target scaler if sidecar explicitly marks target_scaled true'
    $out += '            sc_sidecar = artp / "scaler_sidecar.json"'
    $out += '            use_alt_as_target = False'
    $out += '            if sc_sidecar.exists():'
    $out += '                import json'
    $out += '                meta = json.load(open(sc_sidecar,"r"))'
    $out += '                # expected optional keys: target_scaled (bool) or similar; be conservative'
    $out += '                if meta.get("target_scaled", False):'
    $out += '                    use_alt_as_target = True'
    $out += '            if alt.exists() and use_alt_as_target:'
    $out += '                tsc_p = alt'
    $out += '        if tsc_p.exists():'
    $out += '            tsc = pickle.load(open(tsc_p,"rb"))'
    $out += '            # ensure predictions shape compatible'
    $out += '            try:'
    $out += '                y_pred_unscaled = tsc.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)'
    $out += '                y_pred = y_pred_unscaled'
    $out += '                print("INFO: applied target inverse_transform from", tsc_p)'
    $out += '            except Exception as _e:'
    $out += '                # fallback: try inverse on 1D if transformer supports it'
    $out += '                try:'
    $out += '                    y_pred = tsc.inverse_transform(y_pred)'
    $out += '                    print("INFO: applied target inverse_transform (direct) from", tsc_p)'
    $out += '                except Exception as __e:'
    $out += '                    print("WARN: target inverse_transform failed:", __e)'
    $out += '    except Exception as e:'
    $out += '        # non-fatal: continue scoring with original y_pred'
    $out += '        print("INFO: no target scaler applied or error while applying it:", e)'
    $inserted = $true
  }
}
$out | Set-Content null_tests.py
Write-Output "PATCHED null_tests.py (backup at null_tests.py.bak)."
