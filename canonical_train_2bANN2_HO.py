#!/usr/bin/env python3
import argparse, sys, os, json, hashlib, subprocess, time
from datetime import datetime
import random
import hashlib

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='canonical_train_2bANN2_HO: canonical training entrypoint')
    p.add_argument('--train-file', required=True, help='CSV training inputs')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--epochs', type=int, default=2, help='Epochs for smoke/full runs')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size')
    p.add_argument('--seed', type=int, default=42, help='Deterministic seed')
    p.add_argument('--checkpoint', default=None, help='Optional checkpoint path')
    p.add_argument('--smoke-parse-only', action='store_true', help='Parse-only test (no training)')
    p.add_argument('--smoke-no-train', action='store_true', help='Smoke run without training (checks IO/parsing)')
    return p.parse_args(argv)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def canonicalize_preds(preds):
    # preds: list of floats -> deterministic canonical JSON string
    rounded = [round(float(x), 9) for x in preds]
    # stable ordering: keep input order; convert to JSON single-line
    return json.dumps(rounded, separators=(',',':'), ensure_ascii=False)

def emit_manifest(outdir, manifest):
    os.makedirs(outdir, exist_ok=True)
    mf = os.path.join(outdir, 'run_manifest.txt')
    with open(mf,'w', encoding='utf8') as f:
        for k,v in manifest.items():
            f.write(f"{k}: {v}\n")
    return mf

def main(argv=None):
    args = parse_args(argv)
    start = datetime.utcnow().isoformat() + 'Z'
    # Basic validations
    if not os.path.exists(args.train_file):
        print('ERROR: missing train-file', args.train_file)
        sys.exit(3)
    os.makedirs(args.outdir, exist_ok=True)
    # Deterministic seeding
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass
    # Commit and entrypoint SHA
    commit = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip() if os.path.exists('.git') else 'NO_GIT'
    entry_sha = sha256_file(__file__)
    target_script = os.path.join(os.path.dirname(__file__), 'canonical_train_2bANN2_HO_impl.py')
    target_sha = sha256_file(target_script) if os.path.exists(target_script) else 'MISSING'
    manifest = {
        'run_id': f'canonical_{start}',
        'timestamp': start,
        'runner': 'canonical_train_2bANN2_HO.py',
        'commit_sha': commit,
        'canonical_entrypoint_sha': entry_sha,
        'target_impl_sha': target_sha,
        'train_file_hash': sha256_file(args.train_file),
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'outdir': os.path.abspath(args.outdir)
    }
    # Parse-only smoke
    if args.smoke_parse_only:
        print('PARSE-ONLY OK')
        mf = emit_manifest(args.outdir, manifest)
        print('MANIFEST:', mf)
        sys.exit(0)
    # Smoke no-train: check data read and write permissions
    if args.smoke_no_train:
        with open(args.train_file,'r',encoding='utf8') as f:
            hdr = f.readline().strip()
        test_file = os.path.join(args.outdir, 'smoke_io_ok.txt')
        with open(test_file,'w',encoding='utf8') as tf:
            tf.write('SMOKE_IO_OK\n')
        manifest['smoke_io'] = test_file
        mf = emit_manifest(args.outdir, manifest)
        print('SMOKE NO-TRAIN OK; manifest:', mf)
        sys.exit(0)
    # If target impl exists, delegate; else minimal placeholder training to produce preds
    preds = []
    if os.path.exists(target_script):
        cmd = [sys.executable, target_script, '--train-file', args.train_file, '--outdir', args.outdir, '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--seed', str(args.seed)]
        if args.checkpoint: cmd += ['--checkpoint', args.checkpoint]
        print('DELEGATE:', ' '.join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print('TARGET_IMPL_FAILED rc=', rc); sys.exit(rc)
        # assume target produced preds.json in outdir
        preds_path = os.path.join(args.outdir, 'preds.json')
        if os.path.exists(preds_path):
            with open(preds_path,'r',encoding='utf8') as pf:
                preds = json.load(pf)
    else:
        # Minimal deterministic placeholder training: read csv and output fixed preds
        import csv
        rows = []
        with open(args.train_file,'r',encoding='utf8') as f:
            r = csv.reader(f)
            hdr = next(r, None)
            for row in r:
                rows.append(row)
        # produce naive preds: mean of zeros with small random noise deterministic by seed
        import math
        for i,_ in enumerate(rows):
            preds.append(0.0 + (random.Random(args.seed + i).random() - 0.5) * 1e-6)
        preds_path = os.path.join(args.outdir, 'preds.json')
        with open(preds_path,'w',encoding='utf8') as pf:
            json.dump(preds, pf)
    # canonicalize preds and record SHA
    canonical_preds = canonicalize_preds(preds)
    preds_file = os.path.join(args.outdir, 'preds_canonical.json')
    with open(preds_file,'w',encoding='utf8') as pcf:
        pcf.write(canonical_preds)
    preds_sha = hashlib.sha256(canonical_preds.encode('utf8')).hexdigest()
    manifest['preds_file'] = preds_file
    manifest['preds_sha'] = preds_sha
    mf = emit_manifest(args.outdir, manifest)
    # runner.log header for screen-readers
    rl = os.path.join(args.outdir, 'runner.log')
    with open(rl,'w',encoding='utf8') as rlo:
        rlo.write(f"RUNNER SUMMARY: canonical_train_2bANN2_HO; commit={commit}; entry_sha={entry_sha}\n")
        rlo.write(f"START: {start}\n")
        rlo.write('COMMAND: ' + ' '.join(sys.argv) + '\n')
        rlo.write('MANIFEST: ' + mf + '\n')
    print('COMPLETE; manifest:', mf)
    sys.exit(0)

if __name__=='__main__':
    main()
