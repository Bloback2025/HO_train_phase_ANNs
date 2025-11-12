# append_lineage_runlog_safe.py
import json
import pathlib
import datetime

runlog = pathlib.Path("ho_artifact_outputs") / "RUNLOG_2bANN2_HO_final.json"

# load or start fresh
if runlog.exists():
    data = json.load(open(runlog, "r"))
else:
    data = {}

# ensure notes is a dict; if not, preserve original as notes_orig
notes = data.get("notes")
if isinstance(notes, dict):
    new_notes = dict(notes)  # copy
else:
    new_notes = {}
    if notes is not None:
        new_notes["notes_orig"] = str(notes)

# set required fields (do not overwrite existing explicit target_processing if present)
new_notes.setdefault("target_processing", "none")
new_notes["replaced_from"] = "train_phase2b2_HO.py"
new_notes["lineage_confirmed_at"] = datetime.datetime.utcnow().isoformat() + "Z"

data["notes"] = new_notes

# write back
runlog.parent.mkdir(parents=True, exist_ok=True)
with open(runlog, "w") as f:
    json.dump(data, f, indent=2)

print("UPDATED", str(runlog))
print("notes now contains keys:", list(new_notes.keys()))
