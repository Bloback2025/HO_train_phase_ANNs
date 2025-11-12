#!/usr/bin/env python3
import tempfile, os, json, datetime, shutil, hashlib, sys

def safe_write_runlog(runlog, data):
    try:
        notes = data.get('notes', {}) if isinstance(data, dict) else {}
        notes.setdefault('target_processing', 'none')
        notes['target_processing_written_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
        notes.setdefault('replaced_from', notes.get('replaced_from', 'train_phase2b2_HO.py'))
        data['notes'] = notes

        # deterministic payload and fingerprint
        payload = json.dumps(data, indent=2, sort_keys=True).encode('utf-8')
        fingerprint = hashlib.sha256(payload).hexdigest()

        # atomic write
        dirpath = os.path.dirname(runlog) or '.'
        with tempfile.NamedTemporaryFile('wb', dir=dirpath, delete=False) as tf:
            tf.write(payload)
            tmp_path = tf.name
        shutil.move(tmp_path, runlog)

        # small in-repo index
        idx = os.path.join(dirpath, 'runlog_index.txt')
        with open(idx, 'a', encoding='utf-8') as ix:
            ix.write(f"{datetime.datetime.utcnow().isoformat()}Z {os.path.basename(runlog)} fingerprint={fingerprint}\n")

        print('INFO: target_processing written to', runlog)
        return 0
    except Exception as __e:
        err_line = f"{datetime.datetime.utcnow().isoformat()}Z WARN: failed to write target_processing runlog entry: {__e}\n"
        try:
            sys.stderr.write(err_line)
            err_path = os.path.join(dirpath, 'runlog_write_errors.txt')
            with open(err_path, 'a', encoding='utf-8') as ef:
                ef.write(err_line)
        except Exception:
            pass
        return 1

if __name__ == "__main__":
    # Example invocation for manual test; adjust runlog path and sample data as needed
    sample_runlog = os.path.join(os.path.dirname(__file__), '..', 'ho_artifact_outputs', 'RUNLOG_test_atomic_write.json')
    os.makedirs(os.path.dirname(sample_runlog), exist_ok=True)
    sample_data = {"run_id": "test_atomic", "metrics": {"MSE": 0.0}}
    rc = safe_write_runlog(sample_runlog, sample_data)
    sys.exit(rc)
