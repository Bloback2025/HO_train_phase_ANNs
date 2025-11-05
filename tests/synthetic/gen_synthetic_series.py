#!/usr/bin/env python3
# simple generator for synthetic timestamps (placeholder)
import csv, sys
def gen(n, out):
    with open(out, 'w') as f:
        f.write('timestamp\\n')
        for i in range(int(n)):
            f.write('2025-11-01T00:00:00+00:00\\n')
if __name__ == '__main__':
    gen(sys.argv[1] if len(sys.argv)>1 else 1000, sys.argv[2] if len(sys.argv)>2 else 'sample.csv')
