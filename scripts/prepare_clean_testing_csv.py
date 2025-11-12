# File: prepare_clean_testing_csv.py
#!/usr/bin/env python3
"""
Create two files from hoxnc_testing.csv in the repo root:
- hoxnc_testing_inputs.csv   -> columns: Open,High,Low,Close  (for model input checks/training)
- hoxnc_testing_meta.csv     -> original CSV preserved (includes Date)
Writes to current working directory; no subdirectories assumed.
"""
import pandas as pd
from pathlib import Path
import sys

SRC = Path("hoxnc_testing.csv")
INPUTS_OUT = Path("hoxnc_testing_inputs.csv")
META_OUT = Path("hoxnc_testing_meta.csv")
REQUIRED = ["Open", "High", "Low", "Close"]

if not SRC.exists():
    print(f"ERROR: missing {SRC}", file=sys.stderr)
    sys.exit(2)

df = pd.read_csv(SRC)
df.to_csv(META_OUT, index=False)

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    print("ERROR: missing required columns:", missing, file=sys.stderr)
    sys.exit(3)

inputs_df = df[REQUIRED].copy()
inputs_df.to_csv(INPUTS_OUT, index=False)
print("WROTE:", str(INPUTS_OUT))
print("WROTE:", str(META_OUT))
