Copy-Item 2bANN2.HO.py 2bANN2.HO.py.bak -Force
$append = @"
# ---- audit-safe target_processing writer (appended by maintenance script) ----
try:
    import json, pathlib, datetime
    ART_DIR = pathlib.Path('ho_artifact_outputs')
    ART_DIR.mkdir(parents=True, exist_ok=True)
    runlog = ART_DIR / 'RUNLOG_2bANN2_HO_final.json'
    # load existing runlog or start fresh
    try:
        data = json.load(open(runlog,'r'))
    except Exception:
        data = {}
    notes = data.get('notes')
    if not isinstance(notes, dict):
        notes = {'notes_orig': str(notes) } if notes is not None else {}
    # determine declared target processing
    tp = notes.get('target_processing')
    # If training code set a target_scaler variable in this runtime, persist it; otherwise keep explicit none
    if 'target_scaler' in globals() and globals().get('target_scaler') is not None:
        # attempt to save scaler pickle and sidecar (non-fatal)
        try:
            import pickle
            scaler_path = ART_DIR / 'target_scaler.pkl'
            with open(scaler_path, 'wb') as sf:
                pickle.dump(globals().get('target_scaler'), sf)
            notes['target_processing'] = 'scaled'
            notes['target_scaler_path'] = str(scaler_path)
        except Exception as _e:
            notes.setdefault('target_processing','unknown')
            notes.setdefault('target_processing_error', str(_e))
    else:
        notes.setdefault('target_processing', 'none')
    notes['target_processing_written_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
    notes.setdefault('replaced_from', notes.get('replaced_from','train_phase2b2_HO.py'))
    data['notes'] = notes
    with open(runlog,'w') as f:
        json.dump(data, f, indent=2)
    print('INFO: target_processing written to', runlog)
except Exception as __e:
    # non-fatal; do not interrupt training flow
    print('WARN: failed to write target_processing runlog entry:', __e)
# ---- end appended block ----
"@
Add-Content -Path 2bANN2.HO.py -Value $append -Encoding UTF8
Write-Output "PATCH_APPLIED: 2bANN2.HO.py (backup at 2bANN2.HO.py.bak)"
