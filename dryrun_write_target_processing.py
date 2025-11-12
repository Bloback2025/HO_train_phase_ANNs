import json, pathlib, datetime
ART_DIR = pathlib.Path('ho_artifact_outputs')
ART_DIR.mkdir(parents=True, exist_ok=True)
runlog = ART_DIR / 'RUNLOG_2bANN2_HO_final.json'
try:
    data = json.load(open(runlog,'r')) if runlog.exists() else {}
except Exception:
    data = {}
notes = data.get('notes')
if not isinstance(notes, dict):
    notes = {'notes_orig': str(notes)} if notes is not None else {}
notes.setdefault('target_processing','none')
notes['target_processing_written_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
notes.setdefault('replaced_from','train_phase2b2_HO.py')
data['notes'] = notes
with open(runlog,'w') as f:
    json.dump(data, f, indent=2)
print("DRY-RUN OK: wrote", runlog)
print("notes keys:", list(notes.keys()))
