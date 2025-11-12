import importlib.util, sys
spec = importlib.util.spec_from_file_location('ct', 'canonical_train_2bANN2_HO.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# call parse only to validate flags
mod.parse_args(['--train-file','hoxnc_inputs.csv','--outdir','run_outputs/smoke_parse_test','--smoke-parse-only'])
print('PARSE_TEST_OK')
