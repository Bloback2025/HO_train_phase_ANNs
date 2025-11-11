# HANDOVER for 2bANN2_HO verification

## Summary
- Model: 2bANN2_HO_model.keras
- Runlog: RUNLOG_2bANN2_HO_20251109_212442.json
- Model SHA256: E462009ADC5D763961C89107928B32CBA59586225AC6C97C649F9387C48C9877
- Verification closure: CLOSED:MANIFEST-SHA256:RUNLOG_2bANN2_HO_20251109_212442.json
- Verification timestamp: 2025-11-11T12:03:52.9722185+11:00
- Verification summary file: manifests/verification_summary_RUNLOG_2bANN2_HO_20251109.json
- Canonical manifest: manifests/compliant_manifest_RUNLOG_2bANN2_HO_20251109_212442_verified.json
- Verifier output captured: verifier_all.txt
- Audit ledger entry: audit/manifest_closure.log
- Verification tag: verified/2bANN2_HO_20251109

## Provenance / Steps performed
1. Recreated a compliant manifest and computed model SHA256.
2. Ran verify-manifest-and-sha256.ps1 against the manifest and model; verifier returned PASS with closure token above.
3. Committed manifest, verifier output, verification summary, and appended closure to audit/manifest_closure.log.
4. Pushed commits and tag verified/2bANN2_HO_20251109 to origin.

## Files to inspect for full provenance
- manifests/compliant_manifest_RUNLOG_2bANN2_HO_20251109_212442_verified.json
- manifests/compliant_manifest_runlog_clean.json
- manifests/verification_summary_RUNLOG_2bANN2_HO_20251109.json
- verifier_all.txt
- audit/manifest_closure.log
- git tag: verified/2bANN2_HO_20251109

## Recommended next actions for recipient
- Verify SHA locally: `Get-FileHash -Path "<path-to-model>" -Algorithm SHA256`
- Re-run verifier: `powershell -NoProfile -ExecutionPolicy Bypass -File .\verify-manifest-and-sha256.ps1 -ManifestPath <manifest> -ModelPath <model>`
- Check git tag and files on origin.

## Contact / handoff note
Prepared for handover on 2025-11-11T12:11:52+11:00 by automation following audit-safe procedure. All artifacts are committed and pushed to origin (https://github.com/Bloback2025/ANN_TEST_SUITE.git).
