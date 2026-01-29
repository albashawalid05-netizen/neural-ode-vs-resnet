$ErrorActionPreference = "Stop"
$dataRoot = "data\physionet2012_raw"
$seeds = "0 1 2"
$epochs = 10
$batchSize = 64
$hidden = 128
New-Item -ItemType Directory -Force -Path "results" | Out-Null
New-Item -ItemType Directory -Force -Path "results\tables" | Out-Null
New-Item -ItemType Directory -Force -Path "results\figures" | Out-Null
$cmd = "python -m src.run_physio_suite --data_root $dataRoot --seeds $seeds --epochs $epochs --batch_size $batchSize --hidden $hidden"
Write-Host "Running: $cmd"
Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath "results\physio_run_log.txt"
Write-Host "Done."
Write-Host "Expected outputs:"
Write-Host "  results\tables\physio_summary.csv"
Write-Host "  results\figures\physio_auroc_by_obs_quartile_models.png"
Write-Host "  results\tables\physio_auroc_by_obs_quartile_models.csv"
Write-Host "  results\physio_run_log.txt"
