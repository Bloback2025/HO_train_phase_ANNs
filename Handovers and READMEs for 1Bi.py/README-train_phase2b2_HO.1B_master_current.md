# HANDOVER train_phase2b2_HO.1B Master Document
**Version:** v1.0  
**Date:** 2025-11-26  
**Status:** current  
**Canonical:** true
## Purpose
This master document is the authoritative, dated record for the completed ANN run $runId. It contains the verification, manifest, SHA, smoke test, commit/publish, and troubleshooting steps required to publish the handover artifacts and ensure reproducibility.

**Computed SHA256 for train_phase2b2_HO.1B.py:** 1B19359F619C519D4AD4611E1A6DFCAD40209A5DF2989C6D939E549DF89A07CE

## Canonical and mirror paths
**Canonical path:**  
C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\Handovers and READMEs for 1Bi.py\README-train_phase2b2_HO.1B_master_v1.0_20251126.md
**Mirror path on F:**  
F:\Handovers and READMEs for 1Bi.py\README-train_phase2b2_HO.1B_master_v1.0_20251126.md
## Quick usage (explicit)
1. Inspect the generated manifest at manifests/train_phase2b2_HO.1B.manifest.json and per-file .sha256 files under sha256sums/.  
2. Run the smoke test and confirm expected outputs.  
3. Commit and push the submodule changes (manifests, sha256sums, Handovers) and update the superproject gitlink if/when you choose to publish. (This script does not perform any git commits.)  
4. Verify origin/main exposes HO_train_phase_ANNs/Handovers/HANDOVER-train_phase2b2_HO.1B.json after you update the superproject.
## Full file manifest and brief descriptions
| File or pattern | Purpose | Notes |
|---|---|---|
| 	rain_phase2b2_HO.1B.py | Canonical training/inference script | Compute SHA256 and record in manifest before publishing |
| Handovers/HANDOVER-train_phase2b2_HO.1B.json | Handover metadata and artifact pointers | Must list artifact filenames, URIs, and SHA256 values |
| Handovers/ | Handover artifacts directory | Contains packaged models, sample inputs, outputs, manifests |
| manifests/train_phase2b2_HO.1B.manifest.json | Per-run manifest with file list, sizes, SHA256 | Machine-readable; used by CI and audits |
| sha256sums/*.sha256 | Single-line SHA files for artifacts | Format: <SHA256>  <relative/path> |
| HANDOVER.INDEX.json | Index mapping handovers to manifest SHAs | Update atomically when publishing a new handover |
| equirements.txt | Dependency list | Pin versions for reproducibility |
| un_smoke_test.py | Smoke test runner | Produces Handovers/sample_output/result-train_phase2b2_HO.1B.json |
| configs/*.yaml | Deterministic run configs | Include seeds and dataset pointers |
| checkpoints/*.{pt,h5} | Model weight checkpoints | Prefer external storage; include SHA and size in manifest |
| sample_input/ | Deterministic inputs for smoke test | Small, versioned examples |
| sample_output/ | Expected outputs for smoke test | Used to validate smoke test success |
| ci/verify_handover.ps1 | CI script to compute and validate SHAs | Should fail the build on mismatch |
| ci/push_submodule.ps1 | Script to push submodule and update superproject gitlink | Automates git update-index step |
| .gitmodules | Submodule configuration | Ensure URL and path is correct |
| README-*.md | Dated READMEs and _current pointer | Dated file is canonical; _current is convenience copy |
| ackups/ | Archived intermediate files | Move .bak and temp files here after verification |
## Exact verification and publishing workflow (copy/paste ready)
### 1 Compute SHA256 for primary artifacts
Get-FileHash -Algorithm SHA256 "HO_train_phase_ANNs\train_phase2b2_HO.1B.py"
Get-FileHash -Algorithm SHA256 "HO_train_phase_ANNs\Handovers\HANDOVER-train_phase2b2_HO.1B.json"
### 2 Commit and push submodule (manual step you run when ready)
git -C HO_train_phase_ANNs add manifests Handovers sha256sums HANDOVER.INDEX.json
git -C HO_train_phase_ANNs commit -m "Add manifest and SHAs for train_phase2b2_HO.1B"
git -C HO_train_phase_ANNs push origin main
### 3 Update superproject gitlink and push (manual)
\365004e6c1717c84bd7cb30363d02a36b2a3c0fb = git -C HO_train_phase_ANNs rev-parse HEAD
git update-index --cacheinfo 160000,\365004e6c1717c84bd7cb30363d02a36b2a3c0fb,HO_train_phase_ANNs
git add HO_train_phase_ANNs
git commit -m "Point HO_train_phase_ANNs to \365004e6c1717c84bd7cb30363d02a36b2a3c0fb (include HANDOVER for train_phase2b2_HO.1B)"
git fetch origin
git pull --rebase origin main
git push origin main
### 4 Verify visibility on origin/main
git fetch origin
git ls-tree -r origin/main HO_train_phase_ANNs/Handovers --name-only
git show origin/main:HO_train_phase_ANNs/Handovers/HANDOVER-train_phase2b2_HO.1B.json
## Smoke test reproducibility (explicit commands)
cd "C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_smoke_test.py --config Handovers/smoke_config.yaml --output Handovers/sample_output/
# Expect exit code 0 and file Handovers/sample_output/result-train_phase2b2_HO.1B.json
## Acceptance criteria
- SHA256 values for 	rain_phase2b2_HO.1B.py and HANDOVER-train_phase2b2_HO.1B.json computed and recorded in manifests/ and HANDOVER.INDEX.json.  
- Smoke test runs successfully on a clean environment and produces the expected sample output.  
- This dated master README exists in the canonical folder and is mirrored to F:.
## Troubleshooting
- If Handovers/ is missing, run: git -C HO_train_phase_ANNs submodule update --init --recursive
- If F: is not writable, confirm the drive is mounted and you have write permission.
- If long path errors occur, enable long path support or map a drive letter closer to the repo root.
## Contacts
Owner: Brian (loweb)
Backup contact: (none provided)
