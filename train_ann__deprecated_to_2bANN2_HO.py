# Alias: train_ann__deprecated_to_2bANN2_HO.py
# DEPRECATED_ALIAS: maps legacy calls to canonical_train_2bANN2_HO.py
# CREATED_AT: 20251112_175350
# REMEDIATION_BRANCH: remediation/canonical_train_2bANN2_HO_20251112_175350
# REASON: emergency remediation for missing train_ann.py placeholder
import sys, subprocess, os
target = os.path.join(os.path.dirname(__file__), 'canonical_train_2bANN2_HO.py')
if not os.path.exists(target):
    print('ERROR: canonical target missing', target); sys.exit(2)
cmd = [sys.executable, target] + sys.argv[1:]
print('ALIAS FORWARD:', ' '.join(cmd))
rc = subprocess.call(cmd)
sys.exit(rc)
