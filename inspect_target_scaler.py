import pickle, os, sys
fn = r"ho_artifact_outputs\target_scaler.pkl"
if not os.path.exists(fn):
    print("MISSING target_scaler.pkl, checking scaler.pkl instead")
    fn = r"ho_artifact_outputs\scaler.pkl"
sc = pickle.load(open(fn,"rb"))
print("TYPE:", type(sc))
for a in ("mean_","scale_","var_","min_","data_min_","data_max_"):
    if hasattr(sc,a):
        v = getattr(sc,a)
        print(a, getattr(v, "tolist", lambda: v)() if hasattr(v,"tolist") else v)
print("HAS_INVERSE:", hasattr(sc,"inverse_transform"))
