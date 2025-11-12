# run_2bANN2HO.ps1
param(
    [string]$PythonExe = "python",
    [string]$ScriptPath = "C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\train_phase2b2_HO_reinstated.py",
    [string]$LogDir = "C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\ps_logs"
)

# ensure log dir exists
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile   = Join-Path $LogDir ("run_" + $timestamp + ".log")

Write-Host "Launching deterministic training harness..."
Write-Host "Script: $ScriptPath"
Write-Host "Log: $logFile"

# run python script, capture output
& $PythonExe $ScriptPath *>&1 | Tee-Object -FilePath $logFile -Append | Out-Null

# force file creation if nothing was written
if (!(Test-Path $logFile)) {
    New-Item -ItemType File -Path $logFile | Out-Null
}

# compute hashes
$scriptHash = Get-FileHash -Algorithm SHA256 $ScriptPath | Select-Object -ExpandProperty Hash
$logHash    = Get-FileHash -Algorithm SHA256 $logFile   | Select-Object -ExpandProperty Hash

# record JSON
$hashRecord = @{
    timestamp  = (Get-Date).ToString("o")
    scriptPath = $ScriptPath
    scriptHash = $scriptHash
    logFile    = $logFile
    logHash    = $logHash
}
$hashJson = $hashRecord | ConvertTo-Json -Depth 3
$hashJson | Out-File (Join-Path $LogDir ("hashes_" + $timestamp + ".json")) -Encoding utf8

Write-Host "Run complete. Hashes recorded."
