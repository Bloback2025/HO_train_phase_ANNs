# 2_inspect_validation.py
import pandas as pd

p = r"hoxnc_validation.csv"
df = pd.read_csv(p)

print("=== FIRST 20 ROWS ===")
print(df.head(20).to_csv(index=False).strip())

print("=== SUMMARY ===")
print(f"rows={len(df)}")
# BROKEN: if "y_true" in df.columns:
    s = pd.to_numeric(df["y_true"], errors="coerce")
    print(f"y_true: mean={s.mean():.6f}, std={s.std(ddof=0):.6f}, min={s.min()}, max={s.max()}")
# BROKEN: else:
    print("y_true: MISSING")
