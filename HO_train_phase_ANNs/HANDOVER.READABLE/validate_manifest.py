import json,sys
try:
    with open("HANDOVER.RUN.manifest.json","r",encoding="utf8") as f:
        json.load(f)
    print("JSON_OK")
except Exception as e:
    print("JSON_ERROR:", e)
    raise SystemExit(4)
