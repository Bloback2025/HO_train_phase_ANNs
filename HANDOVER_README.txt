Handoff README: HO 1st time 5080
Updated: 2025-11-09T21:53:20Z
Location: C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\ho_artifact_outputs and C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\ann_stresstest_outputs

Primary RUNLOG (latest):
- RUNLOG_2bANN2_HO_20251109_212442.json  — 35EE767357E1BF46FBAC2B473E51CA588C16EB0C3B17EF4C0859B0CF7192C5CD

Key artifacts and canonical SHA256:
- 2bANN2_HO_model.keras                  — e462009adc5d763961c89107928b32cba59586225ac6c97c649f9387c48c9877
- scaler.pkl                             — FB35F436F83EFF5A8ED27AC46CAF5390521366F391E859C55799EFBCCB16CCDF
- scaler_sidecar.json                    — 0DECB8F6DA8952A4B3040186A401A8D62BEC79132333382BB8835DB06268AE62

Stress-test artifacts (ann_stresstest_outputs):
- reconstructed_ann.keras                — e94a3c535f73708d03d27205cec9fdc9f0456b7c03c3a83f5d6361f82606347c
- stresstest_manifest_1762685480.json    — d955abbc4a54a737841c98ad7031b8099abc156a460532a0b6b32471694c0642
- stresstest_results_1762685480.json     — edf813fda1e6c485a8e8e5334461601d3b90c978fd53936d520469f972685214

Latest run metrics (from RUNLOG_2bANN2_HO_20251109_212442.json):
- MAE  : 0.607881516912376
- RMSE : 0.7485981604448242
- R2   : 0.9960430833235772

Stresstest summary notes:
- Reconstructed model saved untrained; LAYERS 8 PARAMS 16961 (reconstructed topology matched where possible).
- Stresstest produced results and manifest under ann_stresstest_outputs; hashes recorded above and merged into ho_artifact_outputs/artifact_hashes_summary.json.
- Use ho_artifact_outputs/artifact_hashes_summary.json as canonical registry for artifact integrity.

Operational actions:
- To reproduce stress test: run run_ann_stresstest.py in project root (writes to ann_stresstest_outputs).
- To merge new artifacts into canonical registry: use atomic merge procedure in the project ho_artifact_outputs.

