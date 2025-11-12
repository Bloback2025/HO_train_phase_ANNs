import json, pathlib, datetime
runlog = pathlib.Path("ho_artifact_outputs") / "RUNLOG_2bANN2_HO_final.json"
if runlog.exists():
    data = json.load(open(runlog,"r"))
else:
    data = {}
data.setdefault("notes", {})
data["notes"]["target_processing"] = data.get("notes",{}).get("target_processing","none")
data["notes"]["replaced_from"] = "train_phase2b2_HO.py"
data["notes"]["lineage_confirmed_at"] = datetime.datetime.utcnow().isoformat()+"Z"
json.dump(data, open(runlog,"w"), indent=2)
print("UPDATED RUNLOG_2bANN2_HO_final.json")
