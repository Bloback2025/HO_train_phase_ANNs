# File: inspect_columns_min.py
#!/usr/bin/env python3
"""
Minimal column check for hoxnc_testing.csv
- Confirms presence of canonical OHLC inputs only: Open, High, Low, Close
- Reports any extra or missing columns; prints a single SUMMARY line
"""
import sys
import pandas as pd
from pathlib import Path

SRC = Path("hoxnc_testing.csv")
REQUIRED = ["Open", "High", "Low", "Close"]

if not SRC.exists():
    print(f"ERROR: missing {SRC}", file=sys.stderr)
    sys.exit(2)

df = pd.read_csv(SRC)
cols = list(df.columns)
extra = [c for c in cols if c not in REQUIRED]
missing = [c for c in REQUIRED if c not in cols]

print("COLUMNS_LIST:", cols)
print("EXTRA_COLUMNS:", extra)
print("MISSING_COLUMNS:", missing)
print("SUMMARY:", "OK" if (not extra and not missing) else "ALERT")
