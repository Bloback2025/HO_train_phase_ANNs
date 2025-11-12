# train_ann.py - minimal alias -> canonical_train_2bANN2_HO.py
# CREATED: 2025-11-12T21:33:03.8643206+11:00
# REMEDIATION: restores missing legacy entrypoint for reproducible reruns
import sys, subprocess, os
target = os.path.join(os.path.dirname(__file__), 'canonical_train_2bANN2_HO.py')
if not os.path.exists(target):
    print('ERROR: canonical target missing', target); sys.exit(2)
cmd = [sys.executable, target] + sys.argv[1:]
print('ALIAS FORWARD:', ' '.join(cmd))
rc = subprocess.call(cmd)
sys.exit(rc)
