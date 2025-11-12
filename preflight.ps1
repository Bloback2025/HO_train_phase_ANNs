# preflight.ps1 - checks canonical entrypoint presence and records SHA
if (-not (Test-Path './canonical_train_2bANN2_HO.py')) { Write-Error 'Missing canonical_train_2bANN2_HO.py'; exit 2 }
 = (Get-FileHash -Path './canonical_train_2bANN2_HO.py' -Algorithm SHA256).Hash
\"CANONICAL_ENTRYPOINT_SHA: \" | Out-File -FilePath preflight_entry.sha -Encoding utf8
Write-Output 'PREFLIGHT_OK'
