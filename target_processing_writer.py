#!/usr/bin/env python3
import json, pathlib, datetime, tempfile, hashlib, os, pickle, sys, traceback
ART_DIR = pathlib.Path("ho_artifact_outputs"); ART_DIR.mkdir(parents=True, exist_ok=True)
RUNLOG_NAME = "RUNLOG_2bANN2_HO_final.json"; runlog_path = ART_DIR / RUNLOG_NAME
def iso_now_tz(): return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
def sha256_of_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()
def atomic_write_json(obj,target_path):
    d=pathlib.Path(target_path).parent; d.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=str(d),delete=False,encoding="utf-8") as tf:
        json.dump(obj,tf,ensure_ascii=False,sort_keys=True,indent=2); tf.flush(); os.fsync(tf.fileno())
    os.replace(tf.name,target_path)
def load_existing_runlog(path):
    try:
        if path.exists():
            with path.open("r",encoding="utf-8") as rf:
                return json.load(rf) or {}
    except Exception as e:
        return {"__load_error":{"when":iso_now_tz(),"error":str(e),"traceback":traceback.format_exc()}}
    return {}
def main():
    data = load_existing_runlog(runlog_path)
    notes = data.get("notes") or {}
    if not isinstance(notes, dict): notes = {"notes_orig": str(notes)} if notes is not None else {}
    scaler = globals().get("target_scaler", None)
    try:
        if scaler is not None:
            scaler_path = ART_DIR / "target_scaler.pkl"
            try:
                with open(scaler_path,"wb") as sf:
                    pickle.dump(scaler,sf,protocol=pickle.HIGHEST_PROTOCOL); sf.flush(); os.fsync(sf.fileno())
                notes["target_processing"]="scaled"; notes["target_scaler_path"]=str(scaler_path)
                try: notes["target_scaler_sha256"]=sha256_of_file(scaler_path); notes["target_scaler_pickle_protocol"]=pickle.HIGHEST_PROTOCOL
                except Exception as _h: notes.setdefault("target_processing_errors",[]).append({"when":iso_now_tz(),"stage":"scaler_hash","error":str(_h)})
            except Exception as e_save:
                notes.setdefault("target_processing_errors",[]).append({"when":iso_now_tz(),"stage":"save_scaler","error":str(e_save),"traceback":traceback.format_exc()})
                notes["target_processing"]="unknown"
        else:
            notes.setdefault("target_processing","none")
    except Exception as e_outer:
        notes.setdefault("target_processing_errors",[]).append({"when":iso_now_tz(),"stage":"scaler_acquire","error":str(e_outer),"traceback":traceback.format_exc()})
        notes.setdefault("target_processing","unknown")
    notes["target_processing_written_at"]=iso_now_tz()
    notes.setdefault("replaced_from", notes.get("replaced_from","train_phase2b2_HO.py"))
    data["notes"]=notes
    try:
        atomic_write_json(data,str(runlog_path))
        try:
            runlog_sha = sha256_of_file(runlog_path)
            data.setdefault("notes",{})["runlog_written_sha256"]=runlog_sha
            atomic_write_json(data,str(runlog_path))
        except Exception as _h2:
            notes.setdefault("target_processing_errors",[]).append({"when":iso_now_tz(),"stage":"runlog_hash","error":str(_h2)})
        print("INFO: target_processing written to", runlog_path)
    except Exception as __e:
        notes.setdefault("target_processing_errors",[]).append({"when":iso_now_tz(),"stage":"atomic_write_runlog","error":str(__e),"traceback":traceback.format_exc()})
        try:
            with runlog_path.open("w",encoding="utf-8") as f: json.dump(data,f,indent=2)
            print("WARN: fallback non-atomic write succeeded for", runlog_path)
        except Exception as __e2:
            print("ERROR: failed to write target_processing runlog fallback:", __e2)
            sys.exit(2)
if __name__ == "__main__": main()
