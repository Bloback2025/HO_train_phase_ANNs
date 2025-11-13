# ANN_TEST_SUITE
Deterministic Inference will complete this.
2025-11-12: created hoxnc_testing_inputs.csv (Date column removed) for model ingestion; original archived as hoxnc_testing_meta_original.csv; manifest entry and SHA256 recorded in manifest.txt.
2025-11-12: created hoxnc_testing_inputs.csv (Date column removed) for model ingestion; original archived as hoxnc_testing_meta_original.csv; manifest entry and SHA256 recorded in manifest.txt.

## Addendum — Post-backfill audit and closure
Date: 2025-11-13 15:39:42 +11:00

- **Scope:** Automated backfill executed across run_outputs to populate required manifest fields for provenance ingestion.
- **Commit:** remediation/canonical_train_2bANN2_HO_20251112_175716 @ c9f1adc
- **Tag:** preds_canonicalized_20251112_175716 (pushed to origin)
- **Files updated:** 17 run_manifest.txt files under run_outputs; .bak copies preserved for each overwritten manifest.
- **Verification:** Run sweep produced no missing run_manifest files; manifests contain explicit fields: run_id, timestamp, runner, commit_sha, canonical_entrypoint_sha, train_file_path, train_file_hash, seed, epochs, mode, preds_sha.
- **Outstanding actions:** canonicalize any remaining preds.json → preds_canonical.json where required and populate preds_sha in manifests (policy: 9-decimal rounding, compact JSON).
- **Next recommended commits:** add atomic manifest writer helper to entrypoint and optionally script preds canonicalization + manifest update.
- **Audit marker:** closure recorded here following push of remediation branch and annotated tag.

## Addendum — Post-backfill audit and closure
Date: 2025-11-13 15:44:14 +11:00

- Scope: Automated backfill executed across run_outputs to populate required manifest fields for provenance ingestion.
- Commit: remediation/canonical_train_2bANN2_HO_20251112_175716 @ c9f1adc (follow-up commits: c1614fca)
- Tag: preds_canonicalized_20251112_175716 (pushed to origin)
- Files updated: 17 run_manifest.txt files under run_outputs; .bak copies preserved for each overwritten manifest.
- Verification: Run sweep produced no missing run_manifest files; manifests contain explicit fields: run_id, timestamp, runner, commit_sha, canonical_entrypoint_sha, train_file_path, train_file_hash, seed, epochs, mode, preds_sha.
- Outstanding actions: add atomic manifest writer to entrypoint (completed); canonicalize any remaining preds.json → preds_canonical.json where required and populate preds_sha in manifests (policy: 9-decimal rounding, compact JSON).
- Next steps: (1) run preds canonicalization script across run_outputs and update manifests with preds_sha, (2) run final manifest sweep and export provenance JSONL for archival.
- Audit marker: Closure recorded here following push of remediation branch and annotated tag.
