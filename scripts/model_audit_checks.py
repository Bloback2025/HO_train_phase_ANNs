#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
import pickle

CSV = "data/hoxnc_full.csv"
MODEL = "ho_artifact_outputs/2bANN2_HO_model.keras"
SCALER = "models/scaler.pkl"
MIN_ROWS = 50

def fail(msg, code=2):
    print("ERROR:", msg)
    sys.exit(code)

def load_data(csv_path):
    p = Path(csv_path)
    if not p.exists():
        fail(f"csv_missing:{csv_path}")
    df = pd.read_csv(p)
    if "Close_t+1" not in df.columns:
        if "Close" in df.columns:
            df["Close_t+1"] = df["Close"].shift(-1)
            df = df.dropna(subset=["Close_t+1"]).reset_index(drop=True)
        else:
            fail("target_missing:Close_t+1_and_Close")
    return df

def main():
    df = load_data(CSV)
    features = ["Open", "High", "Low", "Close"]
    for f in features:
        if f not in df.columns:
            fail(f"feature_missing:{f}")

    X = df[features].astype(float).values
    y = df["Close_t+1"].astype(float).values
    n = len(X)
    if n < MIN_ROWS:
        fail(f"insufficient_rows:{n}")

    test_n = int(round(0.15 * n))
    val_n = int(round(0.10 * (n - test_n)))
    train_end = n - test_n - val_n
    val_end = train_end + val_n

    X_train = X[:train_end]; X_val = X[train_end:val_end]; X_test = X[val_end:]
    y_train = y[:train_end]; y_val = y[train_end:val_end]; y_test = y[val_end:]

    scaler = None
    scaler_path = Path(SCALER)
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)
        except Exception as e:
            fail(f"scaler_load_failed:{e}")
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)

    X_train_s = scaler.transform(X_train); X_test_s = scaler.transform(X_test)

    model_path = Path(MODEL)
    if not model_path.exists():
        fail(f"model_missing:{MODEL}")
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        fail(f"model_load_failed:{e}")

    try:
        y_pred_train = model.predict(X_train_s, verbose=0).reshape(-1)
        y_pred_test = model.predict(X_test_s, verbose=0).reshape(-1)
    except Exception as e:
        fail(f"model_predict_failed:{e}")

    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    train_r2 = float(r2_score(y_train, y_pred_train))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_r2 = float(r2_score(y_test, y_pred_test))

    print("METRIC: train_mae", train_mae, "train_r2", train_r2)
    print("METRIC: test_mae", test_mae, "test_r2", test_r2)

    try:
        lr = LinearRegression().fit(X_train_s, y_train)
        y_lr_test = lr.predict(X_test_s)
        lr_mae = float(mean_absolute_error(y_test, y_lr_test))
        lr_r2 = float(r2_score(y_test, y_lr_test))
        print("BASELINE: lr_test_mae", lr_mae, "lr_test_r2", lr_r2)
    except Exception as e:
        print("BASELINE_ERROR", e)

    try:
        rng = np.random.RandomState(0)
        y_shuffled = y_train.copy()
        rng.shuffle(y_shuffled)
        tf.keras.backend.clear_session()
        from tensorflow.keras import models, layers
        m = models.Sequential([layers.InputLayer(input_shape=(X_train_s.shape[1],)), layers.Dense(16, activation="relu"), layers.Dense(1)])
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_train_s, y_shuffled, epochs=5, batch_size=64, verbose=0)
        y_shuf_test = m.predict(X_test_s, verbose=0).reshape(-1)
        shuf_mae = float(mean_absolute_error(y_test, y_shuf_test))
        shuf_r2 = float(r2_score(y_test, y_shuf_test))
        print("LABELSHUF: shuf_test_mae", shuf_mae, "shuf_test_r2", shuf_r2)
    except Exception as e:
        print("LABELSHUF_ERROR", e)

if __name__ == "__main__":
    main()
