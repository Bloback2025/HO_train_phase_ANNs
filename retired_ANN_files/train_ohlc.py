import json, argparse, time, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def chronological_split_idx(n, test_frac=0.15, val_frac=0.1):
    test_n = int(round(test_frac * n))
    val_n = int(round(val_frac * (n - test_n)))
    train_end = n - test_n - val_n
    return train_end, train_end + val_n

def build_model(input_dim):
    inp = tf.keras.Input(shape=(input_dim,), name="ohlc_input")
    x = tf.keras.layers.Dense(32, activation="relu")(inp)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)
    m = tf.keras.Model(inputs=inp, outputs=out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse")
    return m

def sha256_of_file(p):
    try:
        b = Path(p).read_bytes()
        import hashlib
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./hoxnc_full.csv")
    ap.add_argument("--manifest", default="./hoxnc_training.with_base_and_lags.manifest.json")
    ap.add_argument("--out_model", default="./models/ohlc_best.keras")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    features = man.get("features", ["Open","High","Low","Close"])
    target = man.get("target", "Close_t+1")

    df = pd.read_csv(args.csv)
    if target not in df.columns and "Close" in df.columns:
        df[target] = df["Close"].shift(-1)
    req = list(features) + [target]
    df = df.dropna(axis=0, subset=req).reset_index(drop=True)

    X = df[features].astype(float).values
    y = df[target].astype(float).values.reshape(-1,1)

    n = len(X)
    if n < 10:
        raise SystemExit("NOT_ENOUGH_ROWS")

    train_end, val_end = chronological_split_idx(n, test_frac=0.15, val_frac=0.1)
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = build_model(input_dim=X_train_s.shape[1])

    outdir = Path(args.out_model)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=str(outdir), monitor="val_loss", save_best_only=True)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    history = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val),
                        epochs=args.epochs, batch_size=args.batch, callbacks=[ckpt, es], verbose=2)

    try:
        model = tf.keras.models.load_model(str(outdir))
    except Exception:
        pass

    y_pred = model.predict(X_test_s).reshape(-1)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    tm = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "manifest_used": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_of_file(str(Path(args.manifest).resolve())),
        "csv_used": str(Path(args.csv).resolve()),
        "csv_sha256": sha256_of_file(str(Path(args.csv).resolve())),
        "model_path": str(Path(args.out_model).resolve()),
        "model_input_dim": int(X_train_s.shape[1]),
        "epochs_ran": len(history.history.get("loss", [])),
        "test_mae": float(mae),
        "test_r2": float(r2)
    }
    pm_path = Path("training_manifests")
    pm_path.mkdir(exist_ok=True)
    p = pm_path.joinpath("training_manifest." + time.strftime("%Y%m%d-%H%M%S") + ".json")
    p.write_text(json.dumps(tm, indent=2), encoding="utf-8")

    print("TRAINING_COMPLETE: model_saved=" + str(Path(args.out_model).resolve()))
    print(f"[METRICS] test_mae={mae:.6f} test_r2={r2:.6f}")

if __name__ == "__main__":
    main()

