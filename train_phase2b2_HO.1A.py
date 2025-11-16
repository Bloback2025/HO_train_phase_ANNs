# train_phase2b2_HO.1A.py
# Deterministic Keras development scaffold created: 2025-11-16T10:29:46
# Derived-from: train_phase2b2_HO.py ; canonical unchanged
import os
import random
import json
import numpy as np
import tensorflow as tf
def set_all_seeds(seed=20251113):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
def load_data_sample():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 10).astype(np.float32)
    y = rng.rand(200,).astype(np.float32)
    return X, y
def build_keras_model(input_dim):
    from tensorflow.keras import layers, models, optimizers, losses, metrics
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=losses.MeanSquaredError(),
                  metrics=[metrics.MeanAbsoluteError()])
    return model
def train_and_save(model, X, y, epochs=2, batch_size=32, seed=20251113):
    set_all_seeds(seed)
    model_path = os.path.join(os.path.dirname(__file__), '2bANN2_HO_model.keras')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_path, save_format='keras')
    return model_path
def produce_preds_and_sidecars(model_path, X):
    preds = tf.keras.models.load_model(model_path).predict(X).flatten().tolist()
    preds_path = os.path.join(os.path.dirname(__file__), 'ho_artifact_outputs', 'preds_canonical.json')
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    with open(preds_path, 'w', encoding='utf8') as f:
        json.dump({"preds": preds}, f, ensure_ascii=False)
    # compute preds sha
    import hashlib
    with open(preds_path, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    preds_sha_path = preds_path + '.sha256.txt'
    with open(preds_sha_path, 'w', encoding='ascii') as f:
        f.write(h)
    return preds_path, preds_sha_path, h
if __name__ == '__main__':
    set_all_seeds()
    X, y = load_data_sample()
    model = build_keras_model(X.shape[1])
    model_path = train_and_save(model, X, y, epochs=1, batch_size=32)
    preds_path, preds_side, preds_sha = produce_preds_and_sidecars(model_path, X)
    print('Scaffold smoke run complete for train_phase2b2_HO.1A.py')
    print('MODEL_FILE=' + model_path)
    print('PREDS_FILE=' + preds_path)
    print('PREDS_SHA=' + preds_sha)
