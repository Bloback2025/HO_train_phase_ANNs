import json, hashlib, platform, sys
from datetime import datetime

# BROKEN: def sha256_file(path):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def rewrite_manifest(
    train_path,
    val_path,
    test_path,
    params,
    metrics,
    debug_checks=None
# BROKEN: ):
    record = {
        "params": params,
        "files": {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
            "test_sha256": sha256_file(test_path),
        },
        "metrics": metrics,
        "debug_checks": debug_checks if debug_checks else {},
        "provenance": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "script": sys.argv[0],
            "python_version": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "torch": __import__("torch").__version__,
        },
        "closure": "sealed"
    }

    manifest_path = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\ho_manifest.json"
# BROKEN:     with open(manifest_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Closure: manifest file rewritten at {manifest_path}")
