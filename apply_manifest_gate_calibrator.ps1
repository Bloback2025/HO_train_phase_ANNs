Copy-Item null_tests.py null_tests.py.bak_safe -Force
$in = Get-Content null_tests.py
$out = @()
foreach ($line in $in) {
  $out += $line
  if ($line -match "Apply target calibrator artifact if present") {
    # replace the unconditional apply block with a manifest-gated block
    $indentMatch = ([regex]::Match($line, "^(\\s*)")).Groups[1].Value
    $indent = $indentMatch
    $block = @(
      $indent + '# Manifest-gated calibrator application: only apply if sidecar explicitly sets "apply_target_calibrator": true',
      $indent + 'try:',
      $indent + '    from pathlib import Path',
      $indent + '    import json, pickle',
      $indent + '    artp = Path(r"ho_artifact_outputs")',
      $indent + '    cal_side = artp / "target_calibrator_sidecar.json"',
      $indent + '    cal_p = artp / "target_calibrator.pkl"',
      $indent + '    apply_cal = False',
      $indent + '    if cal_side.exists():',
      $indent + '        meta = json.load(open(cal_side,"r"))',
      $indent + '        apply_cal = bool(meta.get("apply_target_calibrator", False))',
      $indent + '    if apply_cal and cal_p.exists():',
      $indent + '        cal = pickle.load(open(cal_p,"rb"))',
      $indent + '        try:',
      $indent + '            import numpy as _np',
      $indent + '            y_pred = cal.predict(y_pred.reshape(-1,1)).reshape(-1)',
      $indent + '            print("INFO: applied manifest-approved target calibrator from", cal_p)',
      $indent + '        except Exception as _e:',
      $indent + '            print("WARN: manifest calibrator apply failed:", _e)',
      $indent + 'except Exception:',
      $indent + '    pass'
    )
    $out += $block
    # skip original following lines that attempted unconditional calibrator (best-effort).
    # We assume the original unconditional block is present; leaving it in place would be redundant.
    break
  }
}
# append remaining original file if any (preserve lines after insertion)
$rest = $in[($out.Count) .. ($in.Count -1)]
$out += $rest
$out | Set-Content null_tests.py
Write-Output "PATCHED null_tests.py to require manifest approval for calibrator (backup at null_tests.py.bak_safe)."
