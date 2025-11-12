import pickle, sys, os
fn = r"ho_artifact_outputs\scaler.pkl"
if not os.path.exists(fn):
    print("MISSING", fn); sys.exit(2)
sc = pickle.load(open(fn,"rb"))
print("TYPE:", type(sc))
for a in ("mean_","scale_","var_","min_","data_min_","data_max_"):
    if hasattr(sc,a):
        v = getattr(sc,a)
        print(a, getattr(v, "tolist", lambda: v)() if hasattr(v,"tolist") else v)
# If transformer has transform method, report that too
print("HAS_TRANSFORM:", hasattr(sc,"transform"))
