# create_ann_test_suite.ps1
param(
    [string]$Root = "C:\Users\loweb\AI_Financial_Sims\ANN_TEST_SUITE"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Helper: ensure directory exists
function Ensure-Dir([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
    }
}

# Helper: write file if not exists (don't overwrite)
function Write-IfMissing([string]$path, [string]$content) {
    $dir = Split-Path -Path $path -Parent
    Ensure-Dir $dir
    if (-not (Test-Path -LiteralPath $path)) {
        $content | Out-File -FilePath $path -Encoding UTF8
        Write-Host "Created file: $path"
    } else {
        Write-Host "Skipped (exists): $path"
    }
}

# Root
Ensure-Dir $Root

# Top-level
$dirs = @(
    "$Root\docker",
    "$Root\ci",
    "$Root\ci\tests",
    "$Root\ci\test_helpers",
    "$Root\src",
    "$Root\src\static_analyzer",
    "$Root\src\runtime",
    "$Root\src\utils",
    "$Root\examples",
    "$Root\examples\feature_contracts",
    "$Root\tests",
    "$Root\tests\synthetic",
    "$Root\artifacts",
    "$Root\.github",
    "$Root\.github\workflows"
)

foreach ($d in $dirs) { Ensure-Dir $d }

# Files and minimal content
Write-IfMissing "$Root\README.md" @"
# ANN Test Suite

Workspace root for the ANN test harness.

Path: $Root
"@

Write-IfMissing "$Root\run_manifest.json.template" @"
{
  ""approved_code_hashes"": [],
  ""SANDBOX_IMAGE_SHA"": ""REPLACE_ME"",
  ""SANDBOX_BUILD_RECIPE_SHA"": ""REPLACE_ME"",
  ""seccomp_profile_sha256"": ""REPLACE_ME""
}
"@

# Docker sandbox placeholders
Write-IfMissing "$Root\docker\Dockerfile.sandbox" @"
# Minimal sandbox Dockerfile (placeholder)
FROM python:3.11-slim
WORKDIR /sandbox
COPY runner.py /sandbox/runner.py
CMD [""python3"", ""/sandbox/runner.py""]
"@

Write-IfMissing "$Root\docker\runner.py" @"
# runner.py placeholder for sandbox runner
import sys
print('Sandbox runner placeholder')
sys.exit(0)
"@

Write-IfMissing "$Root\docker\requirements.txt" @"
python-dateutil==2.8.2
cryptography
"@

# CI verifier & seccomp
Write-IfMissing "$Root\ci\verify_approved_hashes.py" @"
# verify_approved_hashes.py placeholder
# Implemented verifier should live here (production script provided separately).
print('verify_approved_hashes placeholder')
"@

Write-IfMissing "$Root\ci\verify_manifest_schema.json" @"
{
  ""type"": ""object"",
  ""properties"": {
    ""approved_code_hashes"": { ""type"": ""array"" },
    ""SANDBOX_IMAGE_SHA"": { ""type"": ""string"" }
  },
  ""required"": [""approved_code_hashes"", ""SANDBOX_IMAGE_SHA""]
}
"@

Write-IfMissing "$Root\ci\seccomp_profile.json" @"
{
  ""defaultAction"": ""SCMP_ACT_ERRNO"",
  ""architectures"": [""SCMP_ARCH_X86_64""],
  ""syscalls"": []
}
"@

# CI tests and helpers
Write-IfMissing "$Root\ci\tests\sandbox_anti_abuse.sh" @"
#!/usr/bin/env bash
echo 'Placeholder sandbox anti-abuse tests'
exit 0
"@
Set-ItemProperty -Path "$Root\ci\tests\sandbox_anti_abuse.sh" -Name IsReadOnly -Value $false
# Make it executable under environments that honor shebang

Write-IfMissing "$Root\ci\tests\test_verifier_negative.py" @"
# pytest placeholder for negative verifier tests
def test_placeholder():
    assert True
"@

Write-IfMissing "$Root\ci\test_helpers\key_fixtures.py" @"
# key_fixtures placeholder
def placeholder():
    return True
"@

Write-IfMissing "$Root\ci\test_helpers\sign_and_write.py" @"
# sign_and_write placeholder
def placeholder():
    return True
"@

# Source: static analyzer, runtime, utils placeholders
Write-IfMissing "$Root\src\static_analyzer\transform_span.py" @"
# transform_span placeholder
def placeholder(): pass
"@

Write-IfMissing "$Root\src\static_analyzer\validators.py" @"
# validators placeholder
def placeholder(): pass
"@

Write-IfMissing "$Root\src\static_analyzer\chunked_spans.py" @"
# chunked_spans placeholder
def placeholder(): pass
"@

Write-IfMissing "$Root\src\runtime\dataframe_proxy.py" @"
# dataframe_proxy placeholder
class DataFrameProxy:
    pass
"@

Write-IfMissing "$Root\src\runtime\validator_errors.py" @"
# validator_errors placeholder
class ValidatorError(Exception):
    def __init__(self, error_code, message, remediation_hint=None, related_fields=None):
        self.error_code = error_code
        super().__init__(message)
"@

Write-IfMissing "$Root\src\utils\iso_ts.py" @"
# iso_ts placeholder
def parse_iso_utc(ts): raise NotImplementedError
"@

Write-IfMissing "$Root\src\utils\intervals.py" @"
# intervals helpers placeholder
def interval_contains_half_open(a,b): return False
"@

# Examples
Write-IfMissing "$Root\examples/sample_timestamps.csv" @"
# timestamp sample CSV placeholder
timestamp
2025-11-01T00:00:00+00:00
"@

Write-IfMissing "$Root\examples/partitions.json" @"
{
  ""train"": [""2025-10-01T00:00:00+00:00"", ""2025-10-15T00:00:00+00:00""],
  ""val"": [""2025-10-15T00:00:00+00:00"", ""2025-10-22T00:00:00+00:00""],
  ""test"": [""2025-10-22T00:00:00+00:00"", ""2025-11-01T00:00:00+00:00""]
}
"@

Write-IfMissing "$Root\examples/feature_contracts/index_mode.json" @"
{
  ""target_alignment_mode"": ""index"",
  ""declared_span_rule"": { ""span_type"": ""fixed_lag"", ""params"": { ""lag_steps"": 5 } }
}
"@

Write-IfMissing "$Root\examples/feature_contracts/time_mode.json" @"
{
  ""target_alignment_mode"": ""time"",
  ""declared_span_rule"": { ""span_type"": ""rolling_window"", ""params"": { ""window_steps"": 10 } }
}
"@

# Tests and synthetic
Write-IfMissing "$Root\tests\test_span_validations.py" @"
# test_span_validations placeholder
def test_placeholder_span():
    assert True
"@

Write-IfMissing "$Root\tests\synthetic\gen_synthetic_series.py" @"
#!/usr/bin/env python3
# simple generator for synthetic timestamps (placeholder)
import csv, sys
def gen(n, out):
    with open(out, 'w') as f:
        f.write('timestamp\\n')
        for i in range(int(n)):
            f.write('2025-11-01T00:00:00+00:00\\n')
if __name__ == '__main__':
    gen(sys.argv[1] if len(sys.argv)>1 else 1000, sys.argv[2] if len(sys.argv)>2 else 'sample.csv')
"@

Write-IfMissing "$Root\tests\synthetic\synthetic_config.yaml" @"
# synthetic config placeholder
n: 1000
chunk_size: 256
"@

# Artifacts placeholders
$artifacts = @(
    "$Root\artifacts\verifier_result.json",
    "$Root\artifacts\provenance_summary.json",
    "$Root\artifacts\spans.json",
    "$Root\artifacts\spans_agg.json",
    "$Root\artifacts\sandbox_anti_abuse.json"
)
foreach ($f in $artifacts) { Write-IfMissing $f "{}" }

# GitHub workflows
Write-IfMissing "$Root\.github\workflows\verifier-smoke.yml" @"
name: Verifier Smoke (placeholder)
on: [push]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Placeholder
        run: echo 'Replace with real workflow steps'
"@

Write-IfMissing "$Root\.github\workflows\strict-smoke.yml" @"
name: Strict Smoke (placeholder)
on: [push]
jobs:
  strict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Placeholder strict
        run: echo 'Replace with strict workflow steps'
"@

Write-Host "Directory structure and placeholder files created under: $Root"
