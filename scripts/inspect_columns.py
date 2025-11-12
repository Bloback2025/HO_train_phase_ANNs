# File: scripts/inspect_columns.py
#!/usr/bin/env python3
"""
Inspect testing CSV for unauthorized/extra columns and simple leakage signatures.
- Allowed input columns (canonical OHLC): Open, High, Low, Close
- Target: Close_{t+1} (computed by shifting Close by -1)
Outputs (printed lines for copy/paste into runlogs):
- COLUMNS_LIST: [...]           (all columns found)
- EXTRA_COLUMNS: [...]          (columns not in allowed set)
- MISSING_COLUMNS: [...]        (allowed columns missing)
- FOUND_TARGET_SHIFT_COLUMN: bool
- EXACT_MATCH_COLS: [...]       (columns exactly equal to Close_t+1 after alignment)
- SHIFTED_EQUALS_TARGET: [...]  (cols whose lagged values equal Close_t+1 for lags 1..3)
- OFFSET_LINEAR_MATCHES: [...] (near-identical linear transforms)
- SUMMARY: OK or ALERT
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PATH = Path("hoxnc_testing.csv")
ALLOWED = ["Open", "High", "Low", "Close"]
LAGS = [1,2,3]

if not PATH.exists():
    print(f"ERROR: Missing file {PATH}", file=sys.stderr)
    sys.exit(2)

df = pd.read_csv(PATH)
cols = list(df.columns)
print("COLUMNS_LIST:", cols)

extra = [c for c in cols if c not in ALLOWED]
missing = [c for c in ALLOWED if c not in cols]
print("EXTRA_COLUMNS:", extra)
print("MISSING_COLUMNS:", missing)

if "Close" not in df.columns:
    print("FOUND_TARGET_SHIFT_COLUMN: False")
    print("EXACT_MATCH_COLS: []")
    print("SHIFTED_EQUALS_TARGET: []")
    print("SUMMARY: ALERT - Close column missing")
    sys.exit(0)

df = df.reset_index(drop=True)
df["Close_t+1"] = df["Close"].shift(-1)
aligned = df.dropna(subset=["Close_t+1"]).reset_index(drop=True)

exact_matches = []
for c in cols:
    if c == "Close_t+1":
        continue
    s = aligned[c].dropna().reset_index(drop=True)
    t = aligned["Close_t+1"].dropna().reset_index(drop=True)
    if len(s) == len(t) and s.equals(t):
        exact_matches.append(c)
print("FOUND_TARGET_SHIFT_COLUMN:", "Close_t+1" in cols)
print("EXACT_MATCH_COLS:", exact_matches)

shifted_equals = []
for c in cols:
    if c == "Close":
        continue
    for lag in LAGS:
        s = df[c].shift(lag).dropna().reset_index(drop=True)
        t = df["Close_t+1"].dropna().reset_index(drop=True)
        if len(s) == len(t) and s.equals(t):
            shifted_equals.append((c, lag))
print("SHIFTED_EQUALS_TARGET:", shifted_equals)

offset_alerts = []
numeric = aligned.select_dtypes(include=[np.number])
if "Close_t+1" in numeric.columns:
    target = numeric["Close_t+1"].values
    for c in numeric.columns:
        if c == "Close_t+1":
            continue
        if len(numeric[c].values) != len(target):
            continue
        a, b = np.polyfit(target, numeric[c].values, 1)
        pred = a * target + b
        rmse = np.sqrt(np.mean((numeric[c].values - pred) ** 2))
        if abs(a - 1.0) < 1e-6 and rmse < 1e-9:
            offset_alerts.append((c, float(a), float(b), float(rmse)))
print("OFFSET_LINEAR_MATCHES:", offset_alerts)

if extra or missing or exact_matches or shifted_equals or offset_alerts:
    print("SUMMARY: ALERT")
else:
    print("SUMMARY: OK")
