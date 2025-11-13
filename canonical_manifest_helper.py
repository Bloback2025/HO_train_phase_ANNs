import os
import hashlib
import datetime
import subprocess

def file_sha256(path):
    try:
        with open(path,'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return 'MISSING'

def git_commit_short():
    try:
        out = subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return 'MISSING'

def write_initial_manifest(outdir, runner_name, entrypoint_path, train_file=None):
    os.makedirs(outdir, exist_ok=True)
    mf = os.path.join(outdir, 'run_manifest.txt')
    tmp = mf + '.tmp'
    entry_sha = file_sha256(entrypoint_path) if os.path.exists(entrypoint_path) else 'MISSING'
    train_hash = file_sha256(train_file) if train_file and os.path.exists(train_file) else 'MISSING'
    # Use UTC ISO format
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = [
        f"run_id: {os.path.basename(outdir)}",
        f"timestamp: {ts}",
        f"runner: {runner_name}",
        f"commit_sha: {git_commit_short()}",
        f"canonical_entrypoint_sha: {entry_sha}",
        f"train_file_path: {train_file or 'MISSING'}",
        f"train_file_hash: {train_hash}",
        "seed: MISSING",
        "epochs: MISSING",
        "mode: MISSING",
        "preds_sha: MISSING"
    ]
    with open(tmp, 'w', encoding='utf8') as fh:
        fh.write(\"\\n\".join(payload))
    os.replace(tmp, mf)
