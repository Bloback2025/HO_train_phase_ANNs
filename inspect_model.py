from tensorflow import keras
m = keras.models.load_model(r'./ho_artifact_outputs/2bANN2_HO_model_copied_20251115_153149.keras')
with open('ho_artifact_outputs/2bANN2_HO_model_metadata.txt','w', encoding='utf8') as f:
    def pf(s): f.write(s + '\n')
    m.summary(print_fn=pf)
    f.write("\nCONFIG:\n")
    f.write(str(m.get_config()))
