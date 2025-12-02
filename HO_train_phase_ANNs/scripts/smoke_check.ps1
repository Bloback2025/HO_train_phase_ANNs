$proj="C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs"
$pred="$proj\deterministic_inference_outputs\preds.json"
if(-not (Test-Path $pred)){ Write-Error "preds.json missing"; exit 2 }
$j = Get-Content $pred -Raw | ConvertFrom-Json
$count = ($j.preds | Measure-Object).Count
if($count -ne 30){ Write-Error "unexpected preds count: $count"; exit 3 }
if($j.preds | Where-Object { [double]::IsNaN([double]$_) -or [double]::IsInfinity([double]$_) }){ Write-Error "non-finite pred value found"; exit 4 }
Write-Output "SMOKE OK: preds_count=$count"; exit 0
