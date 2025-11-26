# HANDOVER train_phase2b2_HO.1B Infrastructure Manifest
**Version:** v1.0
**Date:** 2025-11-26
**Status:** current
**Canonical:** true
Project: HO / AI_Financial_Sims
Submodule: HO_train_phase_ANNs
Handover ID: train_phase2b2_HO.1B
Canonical ANN script: train_phase2b2_HO.1B.py
Summary
This document lists the files and patterns that form the infrastructure for the completed run train_phase2b2_HO.1B. Each entry includes a short description and notes about where the file is used in the pipeline, how provenance is recorded, and what to verify before release.
| File or pattern | Purpose | Notes |
|---|---|---|
| train_phase2b2_HO.1B.py | Canonical ANN training and inference script | Primary code; compute SHA256 before release |
| Handovers/HANDOVER-train_phase2b2_HO.1B.json | Handover metadata and artifact pointers | Must contain artifact filenames, URIs, and SHA256 values |
| Handovers/ | Directory for handover artifacts | Contains packaged models, sample inputs, outputs, manifests |
| HANDOVER.INDEX.json | Index mapping handovers to manifest SHAs | Update atomically when publishing a new handover |
| manifests/train_phase2b2_HO.1B.manifest.json | Per-run manifest with file list, sizes, and SHA256 | Machine readable; used by CI and audits |
| sha256sums/train_phase2b2_HO.1B.sha256 | Single-line SHA file for main artifacts | Format: <SHA256>  <relative/path> |
| requirements.txt | Python dependency list | Pin versions for reproducibility |
| run_smoke_test.py | Smoke test runner for the pipeline | Produces Handovers/sample_output/result-train_phase2b2_HO.1B.json |
| configs/train_phase2b2_HO.1B.yaml | Deterministic run configuration | Includes seeds, dataset pointers, hyperparameters |
| checkpoints/*.pt | Model weight checkpoints | Prefer external storage; include SHA and size in manifest |
| sample_input/ | Deterministic inputs for smoke test | Small, versioned examples |
| sample_output/ | Expected outputs for smoke test | Used to validate smoke test success |
| ci/verify_handover.ps1 | CI script to compute and validate SHAs | Should fail the build on mismatch |
| ci/push_submodule.ps1 | Script to push submodule and update superproject gitlink | Automates git update-index step |
| .gitmodules | Submodule configuration | Ensure URL and path are correct |
| README-*.md | Dated READMEs and _current pointer | Dated file is canonical; _current is convenience copy |
| backups/ | Archived intermediate files | Move .bak and temp files here after verification |
