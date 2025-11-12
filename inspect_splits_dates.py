# inspect_splits_dates.py
# Prints first/last dates for train/val/test to check chronological ordering and overlap
import json
from bootstrap_ho_paths_and_patch import train_path, val_path, test_path
import importlib.util, os

# dynamic import of training module (only for load_csv)
BASE_DIR = __import__("bootstrap_ho_paths_and_patch").BASE_DIR
module_path = os.path.join(BASE_DIR, "train_phase2b2_HO_v5_heavy_v5.1.py")
spec = importlib.util.spec_from_file_location("train_v5_1_module", module_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
load_csv = train_mod.load_csv

dates_train, _, _ = load_csv(train_path)
dates_val, _, _ = load_csv(val_path)
dates_test, _, _ = load_csv(test_path)

out = {
    "train_first": str(dates_train[0]),
    "train_last": str(dates_train[-1]),
    "val_first": str(dates_val[0]),
    "val_last": str(dates_val[-1]),
    "test_first": str(dates_test[0]),
    "test_last": str(dates_test[-1])
}
print(json.dumps(out, indent=2))
