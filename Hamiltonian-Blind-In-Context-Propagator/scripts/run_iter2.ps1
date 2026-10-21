# Idempotent: re-running resumes from runs/<name>/last.pt and skips what is finished.
# Run A (reference) then B (pauli_A), 8000 steps each, same seed.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $PSScriptRoot "..")
$steps = 8000
$runs = @(
  @{ name = "n8_ref";    extra = @() },
  @{ name = "n8_pauliA"; extra = @("model.pauli_A=true") }
)
foreach ($r in $runs) {
  $dir = "runs/$($r.name)"
  $res = "results/$($r.name)/results.json"
  if (Test-Path $res) { continue }
  New-Item -ItemType Directory -Force $dir | Out-Null
  function Get-Step { if (Test-Path "$dir/last.pt") { [int](python -c "import torch;print(torch.load('$dir/last.pt',map_location='cpu',weights_only=False)['step'])") } else { 0 } }
  if ((Get-Step) -lt $steps) {
    $a = @("train.py","--preset","local","--set","train.steps=$steps","train.seed=0","train.out_dir=$dir") + $r.extra
    if (Test-Path "$dir/last.pt") { $a += @("--resume","$dir/last.pt") }
    python @a *>> "$dir/stdout.log"
  }
  if ((Get-Step) -ge $steps) {
    python -m eval.evaluate --ckpt "$dir/last.pt" --data datasets/n8/test.h5 --T 8 16 32 64 --horizon 64 --baselines persistence dmd dmd4 oracle hl_oracle --out "results/$($r.name)" *>> "$dir/eval.log"
  }
}
