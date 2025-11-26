# HANDOVER train_phase2b2_HO.1Bi Master Document
**Version:** v1.0  
**Date:** 2025-11-26  
**Status:** current  
**Canonical:** false

## Purpose

**Computed SHA256 for train_phase2b2_HO.1Bi.py:** 2DA58A6A4AA06E475B0E251E89BF699178C68C7AC763559A179CC88312953FCC

## Canonical and mirror paths
**Canonical path:**
C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\Handovers and READMEs for 1Bi.py\README-train_phase2b2_HO.1Bi_master_v1.0_20251126.md

**Mirror path on F:**
F:\Handovers and READMEs for 1Bi.py\README-train_phase2b2_HO.1Bi_master_v1.0_20251126.md

## Quick usage explicit steps
1. Inspect the generated manifest at manifests/train_phase2b2_HO.1Bi.manifest.json and per-file .sha256 files under sha256sums/.
2. Run the smoke test and confirm expected outputs.
3. When ready to publish, run the git commands in the Publishing section below. This script does not perform git operations.

## Full file manifest and brief descriptions
| File or pattern | Purpose | Notes |
|---|---|---|
| 	rain_phase2b2_HO.1Bi.py | Training and inference script for this run | Compute SHA256 and record in manifest before publishing |
| Handovers/HANDOVER-train_phase2b2_HO.1Bi.json | Handover metadata and artifact pointers | Must list artifact filenames, URIs, and SHA256 values |
| Handovers/ | Handover artifacts directory | Contains packaged models, sample inputs, outputs, manifests |
| manifests/train_phase2b2_HO.1Bi.manifest.json | Per-run manifest with file list, sizes, SHA256 | Machine-readable; used by CI and audits |
| sha256sums/*.sha256 | Single-line SHA files for artifacts | Format: <SHA256>  <relative/path> |
| un_smoke_test.py | Smoke test runner | Produces Handovers/sample_output/result-train_phase2b2_HO.1Bi.json |
| equirements.txt | Dependency list | Pin versions for reproducibility |
| checkpoints/*.{pt,h5} | Model weight checkpoints | Prefer external storage; include SHA and size in manifest |
| ci/ | CI scripts | Automate verification and publishing steps |
| README-*.md | Dated READMEs and _current pointer | Dated file is authoritative for this run |
| ackups/ | Archived intermediate files | Move .bak and temp files here after verification |

## Exact verification and publishing workflow copy/paste ready
### Compute SHA256 for primary artifacts
Get-FileHash -Algorithm SHA256 "HO_train_phase_ANNs\train_phase2b2_HO.1Bi.py"
Get-FileHash -Algorithm SHA256 "HO_train_phase_ANNs\Handovers\HANDOVER-train_phase2b2_HO.1Bi.json"

### Commit and push submodule when ready
git -C HO_train_phase_ANNs add manifests Handovers sha256sums HANDOVER.INDEX.json
git -C HO_train_phase_ANNs commit -m "Add manifest and SHAs for train_phase2b2_HO.1Bi"
git -C HO_train_phase_ANNs push origin main

### Update superproject gitlink and push when ready

## Architecture and artifact contract for synthetic runner
**Runner**  
This run uses the canonical synthetic runner `train_phase2b2_HO.1B.py` (deterministic LinearRegression pipeline). The runner writes artifacts atomically and produces per-file SHA sidecars for audit.
**Artifacts produced by the run**
- `preds_model.json` and `preds_model.json.sha256.txt`
- `promotion_gate_summary.json` and `promotion_gate_summary.json.sha256.txt`
- `run_manifest.<run_id>.patched.json`
- `captured_runs/<run_id>/HANDOVER.CLOSURE.txt`
**Manifest requirements**
- Add each artifact to `manifests/<run_id>.manifest.json` with `path`, `size`, `sha256` (UPPERCASE), and `mtime` (ISO UTC).
- Create single-line `.sha256` files under `sha256sums/` for each primary artifact.
**Smoke test**
- `run_smoke_test.py` must run the script in `synthetic` mode with a fixed seed, verify SHA sidecars, and compare `promotion_gate_summary.json` metrics to expected thresholds (example: `mae <= 0.5`).
**Promotion**
- Promotion to canonical requires: (1) manifest and SHA files present, (2) smoke test passing on a clean environment, (3) human review of `promotion_gate_summary.json`, and (4) updating `HANDOVER.INDEX.json` and the dated master README to mark canonical status.
**Notes**
- Example ANN script SHA computed locally for `train_phase2b2_HO.1Bi.py`: 2DA58A6A4AA06E475B0E251E89BF699178C68C7AC763559A179CC88312953FCC
