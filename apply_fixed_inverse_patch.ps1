Copy-Item null_tests.py null_tests.py.bak2 -Force
$in = Get-Content null_tests.py
$out = @()
$inserted = $false
foreach ($line in $in) {
  $out += $line
  if (-not $inserted -and $line -match "y_pred\s*=\s*model\.predict") {
    $indentMatch = ([regex]::Match($line, "^(\\s*)")).Groups[1].Value
    $indent = $indentMatch
    $b = @(
      $indent + '# If a target scaler exists in artifacts, attempt a safe inverse-transform of predictions',
      $indent + 'try:',
      $indent + '    import pickle',
      $indent + '    from pathlib import Path',
      $indent + '    artp = Path(r"ho_artifact_outputs")',
      $indent + '    sc_sidecar = artp / "scaler_sidecar.json"',
      $indent + '    tsc_p = artp / "target_scaler.pkl"',
      $indent + '    use_alt_as_target = False',
      $indent + '    if sc_sidecar.exists():',
      $indent + '        import json',
      $indent + '        meta = json.load(open(sc_sidecar,"r"))',
      $indent + '        if meta.get("target_scaled", False):',
      $indent + '            use_alt_as_target = True',
      $indent + '    if -not tsc_p.exists() -and use_alt_as_target:',
      $indent + '        alt = artp / "scaler.pkl"',
      $indent + '        if alt.exists():',
      $indent + '            tsc_p = alt',
      $indent + '    if tsc_p.exists():',
      $indent + '        tsc = pickle.load(open(tsc_p,"rb"))',
      $indent + '        try:',
      $indent + '            y_pred_unscaled = tsc.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)',
      $indent + '            y_pred = y_pred_unscaled',
      $indent + '            print("INFO: applied target inverse_transform from", tsc_p)',
      $indent + '        except Exception as _e:',
      $indent + '            try:',
      $indent + '                y_pred = tsc.inverse_transform(y_pred)',
      $indent + '                print("INFO: applied target inverse_transform (direct) from", tsc_p)',
      $indent + '            except Exception as __e:',
      $indent + '                print("WARN: target inverse_transform failed:", __e)',
      $indent + 'except Exception as e:',
      $indent + '    print("INFO: no target scaler applied or error while applying it:", e)'
    )
    $out += $b
    $inserted = $true
  }
}
$out | Set-Content null_tests.py
Write-Output "APPLIED corrected inverse-target patch (backup at null_tests.py.bak2)"
