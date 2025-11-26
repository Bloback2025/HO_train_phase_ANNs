# HANDOVER train_phase2b2_HO.1B Infrastructure Technical Manifest
**Version:** v1.0
**Date:** 2025-11-26
**Status:** current
**Canonical:** true
**Computed SHA256 for train_phase2b2_HO.1B.py:** 1B19359F619C519D4AD4611E1A6DFCAD40209A5DF2989C6D939E549DF89A07CE
Purpose
Machine-friendly manifest for automation, CI, and LLM agents to verify provenance, compute and insert SHA256 values, and ensure the superproject points to the correct submodule commit for train_phase2b2_HO.1B.
Repository layout relative paths
- HO_train_phase_ANNs/
- HO_train_phase_ANNs/train_phase2b2_HO.1B.py
- HO_train_phase_ANNs/Handovers/HANDOVER-train_phase2b2_HO.1B.json
- HO_train_phase_ANNs/manifests/train_phase2b2_HO.1B.manifest.json
- HO_train_phase_ANNs/sha256sums/train_phase2b2_HO.1B.sha256
Verification checklist commands
1. Confirm submodule commit
   git -C HO_train_phase_ANNs rev-parse HEAD
2. List Handovers in commit
   git -C HO_train_phase_ANNs ls-tree -r HEAD --name-only | Select-String "Handovers"
3. Compute SHA256 for main artifacts
   Get-FileHash -Algorithm SHA256 "HO_train_phase_ANNs\train_phase2b2_HO.1B.py"
   Get-FileHash -Algorithm SHA256 "HO_train_phase_ANNs\Handovers\HANDOVER-train_phase2b2_HO.1B.json"
4. Generate manifest JSON
   Use the provided PowerShell snippet to enumerate files, compute SHA256, sizes, and write manifests/train_phase2b2_HO.1B.manifest.json
5. Commit and push submodule
   git -C HO_train_phase_ANNs add manifests Handovers HANDOVER.INDEX.json
   git -C HO_train_phase_ANNs commit -m "Add manifest and SHAs for train_phase2b2_HO.1B"
   git -C HO_train_phase_ANNs push origin main
6. Update superproject gitlink
   =git -C HO_train_phase_ANNs rev-parse HEAD
   git update-index --cacheinfo 160000,,HO_train_phase_ANNs
   git add HO_train_phase_ANNs
   git commit -m "Point HO_train_phase_ANNs to "
   git push origin main
7. Verify visibility on origin/main
   git fetch origin
   git ls-tree -r origin/main HO_train_phase_ANNs/Handovers --name-only
Notes on provenance and storage
- Always compute SHA256 locally and store it in manifests and HANDOVER.INDEX.json
- For large checkpoints, store external URI and SHA256 in manifest
- Keep manifests and sha256sums under version control in the submodule
