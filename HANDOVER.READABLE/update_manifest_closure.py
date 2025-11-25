import json,os,subprocess
m_path = "HANDOVER.RUN.manifest.json"
with open(m_path, "r", encoding="utf8") as f:
    manifest = json.load(f)
if "closures" not in manifest or not isinstance(manifest["closures"], list) or len(manifest["closures"])==0:
    print("ERROR: no closures array or no closure entries")
    raise SystemExit(2)
last = manifest["closures"][-1]
last.setdefault("artifacts", {})
# Update the SHA below if you need a different value
last["artifacts"]["preds_sha256"] = "d837f53ef87def3b355f14ae6d719e4b8a84bfc74514affb5e137605c0d0af43"
try:
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=".", text=True).strip()
    last.setdefault("provenance", {})
    last["provenance"]["commit"] = commit
except Exception:
    pass
tmp = m_path + ".new"
with open(tmp, "w", encoding="utf8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
os.replace(tmp, m_path)
print("UPDATED_MANIFEST")
