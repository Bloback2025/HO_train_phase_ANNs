import io,re
fname="deterministic_inference.py"
s=io.open(fname,"r",encoding="utf-8").read()
s=s.replace("`n","\n")
s=re.sub(r"\n?npred\s*=.*\n","\n",s)
s=re.sub(r"\n?#\s*Audit linear adjust.*?(?:\n.*(pred|y_pred).*\n){0,3}","\n",s,flags=re.S)
io.open(fname,"w",encoding="utf-8").write(s)
print("[OK] cleaned deterministic_inference.py")
