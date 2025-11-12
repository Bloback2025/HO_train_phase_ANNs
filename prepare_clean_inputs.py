# File: prepare_clean_inputs.py
#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

SRC = Path("hoxnc_testing.csv")
OUT = Path("hoxnc_testing_inputs.csv")
REQUIRED = ["Open", "High", "Low", "Close"]

if not SRC.exists():
    print(f"ERROR: missing {SRC}", file=sys.stderr); sys.exit(2)

df = pd.read_csv(SRC)
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    print("ERROR: missing required columns:", missing, file=sys.stderr); sys.exit(3)

df[REQUIRED].to_csv(OUT, index=False)
print("WROTE:", str(OUT))
