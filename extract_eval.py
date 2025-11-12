import json, os, sys, numpy as np

RUNLOG = r"ho_artifact_outputs\RUNLOG_2bANN2_HO_20251109_212442.json"
out_x = "X_eval.npy"
out_y = "y_eval.npy"

with open(RUNLOG, "r") as f:
    runlog = json.load(f)

# Try common keys and artifact locations
def try_save(Xarr, Yarr):
    X = np.array(Xarr)
    y = np.array(Yarr)
    np.save(out_x, X); np.save(out_y, y)
    print("SAVED", out_x, out_y)
    return True

# 1) embedded arrays
if "X_eval" in runlog and "y_eval" in runlog:
    try_save(runlog["X_eval"], runlog["y_eval"]); sys.exit(0)

# 2) direct paths in runlog
for k in ("X_eval_path","x_eval_path","X_eval.npy","x_eval.npy"):
    p = runlog.get(k)
    if isinstance(p, str) and os.path.exists(p):
        X = np.load(p)
        # try to find y next to it
        yguess = os.path.join(os.path.dirname(p), "y_eval.npy")
        if os.path.exists(yguess):
            y = np.load(yguess)
            np.save(out_x, X); np.save(out_y, y)
            print("SAVED from path", p, yguess); sys.exit(0)

# 3) artifacts section
for k in ("artifacts","outputs","files","artifacts_outputs"):
    if k in runlog and isinstance(runlog[k], dict):
        m = runlog[k]
        for name in ("X_eval.npy","X_eval","x_eval.npy","x_eval"):
            if name in m:
                xref = m[name]
                if isinstance(xref, str) and os.path.exists(xref):
                    X = np.load(xref)
                    ypath = m.get("y_eval.npy") or m.get("y_eval") or os.path.join(os.path.dirname(xref), "y_eval.npy")
                    if isinstance(ypath, str) and os.path.exists(ypath):
                        y = np.load(ypath)
                        np.save(out_x, X); np.save(out_y, y)
                        print("SAVED from artifacts", xref, ypath); sys.exit(0)

# 4) search project tree for likely candidates (fallback)
for root, dirs, files in os.walk("."):
    for fn in files:
        if fn.lower() in ("x_eval.npy","x_eval.npy") or fn.lower().startswith("x_eval"):
            xp = os.path.join(root, fn)
            yp = os.path.join(root, "y_eval.npy")
            if os.path.exists(yp):
                X = np.load(xp); y = np.load(yp)
                np.save(out_x, X); np.save(out_y, y)
                print("SAVED from search", xp, yp); sys.exit(0)

print("ERROR: Could not locate X_eval/y_eval in RUNLOG or project tree. Provide paths or save X_eval/y_eval as .npy in project root.")
sys.exit(2)
