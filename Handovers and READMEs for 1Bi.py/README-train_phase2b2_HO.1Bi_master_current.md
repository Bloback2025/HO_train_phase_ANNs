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
