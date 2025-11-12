import json, os, sys, numpy as np

candidates = [
    r"ho_artifact_outputs\deterministic_inference_result_null_target.json",
    r"ho_artifact_outputs\deterministic_inference_result_null_features.json"
]

out_x = "X_eval.npy"
out_y = "y_eval.npy"

def try_write(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    np.save(out_x, X)
    np.save(out_y, y)
    print("SAVED", out_x, out_y)
    return True

for path in candidates:
    if not os.path.exists(path):
        continue
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        print("WARN cannot parse", path, e)
        continue

    # common key names for inputs and targets
    x_keys = ["X","X_eval","inputs","features","x","x_eval","input"]
    y_keys = ["y","y_eval","y_true","target","targets","labels","label","y_actual"]

    # direct arrays
    for k in x_keys:
        if k in j and isinstance(j[k], (list,)):
            for ky in y_keys:
                if ky in j and isinstance(j[ky], (list,)):
                    try:
                        if try_write(j[k], j[ky]):
                            sys.exit(0)
                    except Exception as e:
                        print("WARN write failed", e)

    # nested results: look for "results","deterministic","inference","predictions" containers
    for container in ("results","deterministic","inference","inference_results","output"):
        if container in j and isinstance(j[container], dict):
            sub = j[container]
            for k in x_keys:
                if k in sub and isinstance(sub[k], (list,)):
                    for ky in y_keys:
                        if ky in sub and isinstance(sub[ky], (list,)):
                            try:
                                if try_write(sub[k], sub[ky]):
                                    sys.exit(0)
                            except Exception as e:
                                print("WARN nested write failed", e)

    # look for keys that contain arrays anywhere in the json (shallow search)
    flat_keys = {k:v for k,v in j.items() if isinstance(v,(list,))}
    for k,v in flat_keys.items():
        if any(tok in k.lower() for tok in ("x","input","feature")):
            for ky,yv in flat_keys.items():
                if any(tok in ky.lower() for tok in ("y","target","label")):
                    try:
                        if try_write(v, yv):
                            sys.exit(0)
                    except Exception as e:
                        print("WARN flat write failed", e)

print("ERROR: extractor did not find matching X/y arrays in deterministic files. Provide sample file content if needed.")
sys.exit(2)
