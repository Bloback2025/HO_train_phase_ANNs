# env_check.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$report = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    python = [ordered]@{}
    packages = [ordered]@{}
    ok = $true
}

# Ensure artifacts dir
$artifacts = Join-Path (Get-Location) "artifacts"
if (-not (Test-Path $artifacts)) { New-Item -ItemType Directory -Path $artifacts | Out-Null }

# Find python executable
$pythonCmd = $null
$possible = @("python","python3","py")
foreach ($p in $possible) {
    try {
        $ver = & $p -c "import sys; print(sys.version.split()[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) { $pythonCmd = $p; break }
    } catch {}
}
if (-not $pythonCmd) {
    $report.python.found = $false
    $report.python.msg = "No python executable found on PATH (tried: python, python3, py)"
    $report.ok = $false
} else {
    $report.python.found = $true
    $ver = (& $pythonCmd -c "import sys; print(sys.version.split()[0])").Trim()
    $report.python.executable = $pythonCmd
    $report.python.version = $ver
    try {
        $parts = $ver.Split('.')
        $major = [int]$parts[0]; $minor = [int]$parts[1]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            $report.python.sufficient = $false
            $report.python.msg = "Python version must be >= 3.8"
            $report.ok = $false
        } else {
            $report.python.sufficient = $true
        }
    } catch {
        $report.python.sufficient = $false
        $report.python.msg = "Could not parse Python version"
        $report.ok = $false
    }
}

# Helper to run python import checks
function PyCheck([string]$expr) {
    try {
        $out = & $pythonCmd -c $expr 2>&1
        if ($LASTEXITCODE -ne 0) { return @{ ok = $false; out = $out } }
        return @{ ok = $true; out = $out }
    } catch {
        return @{ ok = $false; out = $_.Exception.Message }
    }
}

if ($report.python.found -eq $true) {
    # hashlib (stdlib) check
    $h = PyCheck "import hashlib; print('ok')"
    $report.packages.hashlib = if ($h.ok) { @{ installed = $true } } else { $report.ok = $false; @{ installed = $false; error = $h.out } }

    # numpy
    $n = PyCheck "import importlib, json; m=importlib.import_module('numpy'); print(m.__version__)"
    $report.packages.numpy = if ($n.ok) { @{ installed = $true; version = $n.out.Trim() } } else { $report.ok = $false; @{ installed = $false; error = $n.out } }

    # pandas
    $p = PyCheck "import importlib; m=importlib.import_module('pandas'); print(m.__version__)"
    $report.packages.pandas = if ($p.ok) { @{ installed = $true; version = $p.out.Trim() } } else { $report.ok = $false; @{ installed = $false; error = $p.out } }

    # torch
    $t = PyCheck "import importlib, json; m=importlib.import_module('torch'); v=getattr(m,'__version__',None); cuda_avail = getattr(m,'cuda',None) is not None and m.cuda.is_available(); print(json.dumps({'ver':v,'cuda':cuda_avail}))"
    if ($t.ok) {
        try {
            $parsed = $t.out.Trim() | ConvertFrom-Json
            $report.packages.torch = @{ installed = $true; version = $parsed.ver; cuda_available = $parsed.cuda }
            # Determine CPU-only inference: prefer cuda_available == False
            if ($parsed.cuda -eq $true) {
                $report.packages.torch.cpu_only = $false
                $report.ok = $false
            } else {
                $report.packages.torch.cpu_only = $true
            }
        } catch {
            $report.packages.torch = @{ installed = $true; info = $t.out.Trim() }
        }
    } else {
        $report.packages.torch = @{ installed = $false; error = $t.out }
        $report.ok = $false
    }
}

# Write report to artifacts/env_check.json
$outPath = Join-Path $artifacts "env_check.json"
($report | ConvertTo-Json -Depth 5) | Out-File -FilePath $outPath -Encoding UTF8

# Print concise summary
if ($report.ok) {
    Write-Host "ENV CHECK: OK"
} else {
    Write-Host "ENV CHECK: FAIL"
}
Write-Host "Report written to: $outPath"
# Also display the JSON summary
Get-Content $outPath -Raw
