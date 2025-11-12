from tensorflow.keras.models import load_model
import json, sys
m = load_model(r"ho_artifact_outputs\2bANN2_HO_model.keras")
print("MODEL_SUMMARY_START")
m.summary()
print("MODEL_SUMMARY_END")
try:
    ish = m.input_shape
    print("INPUT_SHAPE:", ish)
except Exception as e:
    try:
        print("INPUT_SHAPE_FROM_INPUTS:", [i.shape for i in m.inputs])
    except Exception as e2:
        print("INPUT_SHAPE: unknown", e, e2)
# also print a minimal list of top layers
print("LAYER_NAMES:", [layer.name for layer in m.layers[:8]])
