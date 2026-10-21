# Waits for the evaluation of each run and publishes it on the dashboard (idempotent: .published marker).
# Export A contains only run A; export B contains A and B (so the dashboard can compare them).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $PSScriptRoot "..")
$jobs = @(
  @{ name = "n8_ref";    keep = @("n8_ref");              label = "n8 A riferimento (no probe, no opzioni coppie, S off)";
     notes = "Run A: riferimento. pair_attn/triangle/pair_latent spenti, nessun L_probe (lambda_probe=0, nessun modulo probe), use_S=false, n=8, 8000 step su CPU, seed 0, validazione ogni 1000 step. Lettura di G a posteriori: AUROC, corr con J e con h." },
  @{ name = "n8_pauliA"; keep = @("n8_ref", "n8_pauliA"); label = "n8 B pauli_A (un generatore di Pauli per tutto A)";
     notes = "Run B: come A ma model.pauli_A=true: tutti i blocchi A_ij (anche diagonali) da un solo MLP come somma di una base di Pauli fissa (m=32, q=5, peso<=2, 106 termini). Stesso seed e stessi step di A. L'export contiene anche A per il confronto." }
)
foreach ($j in $jobs) {
  $marker = "results/$($j.name)/.published"
  while (-not ((Test-Path "results/$($j.name)/results.json") -or (Test-Path $marker))) { Start-Sleep -Seconds 300 }
  if (Test-Path $marker) { continue }
  $st = "stage_publish"
  Remove-Item -Recurse -Force $st -ErrorAction SilentlyContinue
  foreach ($k in $j.keep) {
    New-Item -ItemType Directory -Force "$st/runs/$k", "$st/results/$k" | Out-Null
    Copy-Item "runs/$k/config.json", "runs/$k/log.jsonl", "runs/$k/last.pt" "$st/runs/$k"
    Copy-Item "results/$k/results.json" "$st/results/$k"
  }
  python -m tools.publish_dashboard --label $j.label --notes $j.notes --runs "$st/runs" --results "$st/results" --data datasets/n8/test.h5 --ckpt "runs/$($j.name)/last.pt" --T 32 --horizon 64 --sample 0 --keep 100 *>> "results/publish.log"
  if ($LASTEXITCODE -eq 0) { New-Item -ItemType File $marker | Out-Null }
  Remove-Item -Recurse -Force $st -ErrorAction SilentlyContinue
}
