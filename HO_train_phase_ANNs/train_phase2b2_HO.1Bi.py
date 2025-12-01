#!/usr/bin/env python3
# train_phase2b2_HO.1Bi.py - wrapper: synthetic smoke run by default; forwards full mode to canonical trainer
# Parent SHA256: 1B19359F619C519D4AD4611E1A6DFCAD40209A5DF2989C6D939E549DF89A07CE
import argparse, json, hashlib, tempfile, os, sys, subprocess, shlex
from pathlib import Path
from datetime import datetime

# Paths (adjust if your layout differs)
TRAINER_PATH = Path(r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\train_phase2b2_HO.1B.py")
DATASET_DIR = Path(r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080")
OUTDIR_BASE = Path(r"C:\Users\loweb\AI_Financial_Sims\HO\captured_runs")
PARENT_SHA = "1B19359F619C519D4AD4611E1A6DFCAD40209A5DF2989C6D939E549DF89A07CE"
DEFAULT_SEED = 20251117
TSFMT = "%Y%m%d_%H%M%S"

def sha256_upper(path: Path):
    import hashlib
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest().upper()

def write_atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, os
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tf:
        tf.write(data); tf.flush(); os.fsync(tf.fileno())
    os.replace(tf.name, str(path))

def synthetic_smoke_run(outdir: Path, seed: int):
    outdir.mkdir(parents=True, exist_ok=True)
    preds = {"run_id": outdir.name, "seed": seed, "preds":[{"id":"SYNTH_1","score":0.1234}]}
    preds_bytes = json.dumps(preds, indent=2).encode("utf-8")
    preds_path = outdir / "preds_model.json"; write_atomic(preds_path, preds_bytes)
    preds_sha = hashlib.sha256(preds_bytes).hexdigest().upper()
    (outdir / "preds_model.json.SHA256.TXT").write_text(preds_sha + "\n", encoding="ascii")
    manifest = {"run_id": outdir.name, "timestamp": datetime.utcnow().isoformat()+"Z", "mode":"synthetic", "trainer_parent_sha": PARENT_SHA}
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    (outdir / f"run_manifest.{outdir.name}.patched.json").write_bytes(manifest_bytes)
    (outdir / "HANDOVER.CLOSURE.txt").write_text(f"CLOSURE {outdir.name} {datetime.utcnow().isoformat()} SEAL\n", encoding="utf-8")
    return {"preds_path": str(preds_path), "preds_sha": preds_sha}

def forward_to_trainer(dataset: Path, seed: int, extra_args: list):
    if not TRAINER_PATH.exists():
        sys.stderr.write(f"Missing trainer: {TRAINER_PATH}\n"); return 2
    cmd = ["python", str(TRAINER_PATH), "--dataset", str(dataset), "--seed", str(seed)] + extra_args
    print("Forwarding to canonical trainer:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.stderr.write(f"Canonical trainer failed rc={rc}\n")
    return rc

def parse_args():
    p=argparse.ArgumentParser(description="1Bi wrapper: synthetic or forward full to canonical trainer")
    p.add_argument("--mode", choices=["synthetic","full"], default="synthetic")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    p.add_argument("extra", nargs=argparse.REMAINDER)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "synthetic":
        ts = datetime.utcnow().strftime(TSFMT)
        run_id = f"run_{ts}"
        outdir = OUTDIR_BASE / run_id
        summary = synthetic_smoke_run(outdir, args.seed)
        print(json.dumps({"status":"ok","run_id": run_id, "artifacts": summary}, indent=2))
        return 0
    else:
        # forward all remaining args to canonical trainer; ensure dataset and seed are passed
        extra = args.extra if args.extra else []
        return forward_to_trainer(Path(args.dataset), args.seed, extra)

if __name__ == "__main__":
    sys.exit(main())

# staged-marker
