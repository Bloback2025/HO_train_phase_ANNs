import json, pathlib
p = pathlib.Path("ho_artifact_outputs") / "target_calibrator_sidecar.json"
if not p.exists():
    raise SystemExit("MISSING sidecar; create or run save_target_calibrator.py first")
meta = json.load(open(p,"r"))
meta["apply_target_calibrator"] = True
meta["approved_by"] = "repo_maintainer"
meta["approved_at"] = __import__("datetime").datetime.utcnow().isoformat()+"Z"
json.dump(meta, open(p,"w"), indent=2)
print("ENABLED calibrator in", p)
