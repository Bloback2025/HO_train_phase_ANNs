import os

# --- Define base directory once ---
BASE_DIR = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080"

# --- Construct canonical paths ---
train_path = os.path.join(BASE_DIR, "hoxnc_training.csv")
val_path   = os.path.join(BASE_DIR, "hoxnc_validation.csv")
test_path  = os.path.join(BASE_DIR, "hoxnc_testing.csv")

# --- Optional: quick integrity check ---
# BROKEN: def check_files_exist():
# BROKEN:     for f in [train_path, val_path, test_path]:
# BROKEN:         if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

# Run check immediately if imported
check_files_exist()
