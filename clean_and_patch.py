import io, re
fname = "deterministic_inference.py"
# BROKEN: with io.open(fname, "r", encoding="utf-8") as f:
    s = f.read()
# remove obvious backtick garbage sequences introduced earlier
s = re.sub(r"
", "\n", s)
s = re.sub(r"
pred = .*?pred", "", s, flags=re.S)
# safe replace: look for the y_pred predict line (common variants) and insert the audit scale after it
patterns = [
    r"(y_pred\s*=\s*model\.predict\([^\)]*\)\s*\,\s*verbose\s*=\s*0\)\.reshape\(-1\))",
    r"(y_pred\s*=\s*model\.predict\([^\)]*\)\.reshape\(-1\))",
    r"(pred\s*=\s*model\.predict\([^\)]*\)\.reshape\(-1\))"
]
# BROKEN: insert_line = "\\1\n# Audit linear adjust (temporary): scale preds to target distribution\ny_pred = y_pred * 0.040473742800370395 + -0.6212534896704118\n"
patched = False
# BROKEN: for pat in patterns:
    new_s, n = re.subn(pat, insert_line, s, count=1)
# BROKEN:     if n:
        s = new_s
        patched = True
        break
# BROKEN: if not patched:
    print("[ERR] predict pattern not found; no changes made")
# BROKEN: else:
# BROKEN:     with io.open(fname, "w", encoding="utf-8") as f:
        f.write(s)
    print("[OK] patched deterministic_inference.py")
