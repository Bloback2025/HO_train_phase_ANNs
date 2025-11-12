#!/usr/bin/env python3
# canonical_train_2bANN2_HO_impl.py - minimal deterministic impl used for smoke.
import argparse, json, csv, os, random
def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--train-file', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()
def main():
    args = parse()
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    with open(args.train_file,'r',encoding='utf8') as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            rows.append(row)
    preds = []
    for i,_ in enumerate(rows):
        preds.append(0.0 + (random.Random(args.seed + i).random() - 0.5)*1e-6)
    with open(os.path.join(args.outdir,'preds.json'),'w',encoding='utf8') as pf:
        json.dump(preds, pf)
    print('IMPL COMPLETE', os.path.join(args.outdir,'preds.json'))
if __name__=='__main__':
    main()
