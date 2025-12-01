# scripts/run_smoke.py
# Minimal deterministic smoke runner used for remediation
import json,sys,os,hashlib
repo = os.path.abspath(os.path.dirname(__file__) + os.sep + "..")
manifest_path = os.path.join(repo, "HANDOVER.RUN.manifest.json")
out_dir = os.path.join(repo, "deterministic_inference_outputs")
os.makedirs(out_dir, exist_ok=True)
# deterministic payload (replace with real inference later)
preds = {"generated":"REMEDIATION_PLACEHOLDER","timestamp":None,"preds":[{"id":1,"value":0.0}]}
try:
    # read manifest if present
    if os.path.exists(manifest_path):
        with open(manifest_path,"r",encoding="utf8") as f:
            m = json.load(f)
    else:
        m = {}
    preds["timestamp"] = __import__("datetime").datetime.utcnow().isoformat()+"Z"
    preds_path = os.path.join(out_dir,"preds.json")
    with open(preds_path,"w",encoding="utf8") as f:
        json.dump(preds,f,indent=2,ensure_ascii=False)
    # compute sha
    h = hashlib.sha256()
    with open(preds_path,"rb") as f:
        h.update(f.read())
    sha = h.hexdigest()
    # write simple log to stdout for automation capture
    print("SMOKE_RUN: wrote preds at", preds_path)
    print("SMOKE_RUN: preds_sha256", sha)
    sys.exit(0)
except Exception as e:
    print("SMOKE_RUN_EXCEPTION:", str(e), file=sys.stderr)
    sys.exit(2)
