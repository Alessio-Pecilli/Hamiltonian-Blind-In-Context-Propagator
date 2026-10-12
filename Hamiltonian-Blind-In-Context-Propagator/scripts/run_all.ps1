# Full local pipeline on Windows (PowerShell), from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Optional:  -Steps 20000  -Workers 4  -SkipInstall
#            -Publish [-Label "n8 no S"] [-Notes "..."]   publish the dashboard on GitHub Pages at the end
param(
    [int]$Steps = 20000,
    [int]$Workers = 4,
    [switch]$SkipInstall,
    [switch]$Publish,
    [string]$Label = "run_all",
    [string]$Notes = ""
)
$ErrorActionPreference = "Stop"
function Run($cmd) {
    Write-Host "`n>>> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" }
}

if (-not $SkipInstall) { Run "python -m pip install -r requirements.txt" }
Run "python -c `"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())`""

# 1. unit tests (Sec. 8)
Run "python -m pytest -q tests"

# 2. datasets n = 8 (Sec. 2.6); test set longer so k can reach 64
$G = "python -m data.generate --family tfim --dt 0.1 --workers $Workers"
if (-not (Test-Path datasets\n8\train.h5)) { Run "$G --n 8 --ntraj 1000 --L_traj 160 --seed 1 --out datasets/n8/train.h5" }
if (-not (Test-Path datasets\n8\val.h5))   { Run "$G --n 8 --ntraj 100 --L_traj 160 --seed 2 --out datasets/n8/val.h5" }
if (-not (Test-Path datasets\n8\test.h5))  { Run "$G --n 8 --ntraj 100 --L_traj 192 --seed 3 --out datasets/n8/test.h5" }

# 3. Stage 1 check (i): single-batch overfit to L_pred < 1e-4
Run "python train.py --preset local --overfit --set train.steps=10000 train.out_dir=runs/overfit data.num_workers=0"

# 4. Stage 1 training (Head A)
Run "python train.py --preset local --set train.steps=$Steps train.out_dir=runs/local_A data.num_workers=0"

# 5. evaluation + baselines (Sec. 6-7)
Run "python -m eval.evaluate --ckpt runs/local_A/last.pt --data datasets/n8/test.h5 --T 8 16 32 64 --horizon 64 --baselines persistence dmd dmd4 oracle hl_oracle --out results/n8_A"
# 6. optional: export + commit + push the dashboard (docs/data in the git root, served by GitHub Pages)
if ($Publish) {
    Run "python -m tools.publish_dashboard --label '$Label' --notes '$Notes' --runs runs --results results --data datasets/n8/test.h5 --ckpt runs/local_A/last.pt --T 32 --horizon 64 --sample 0"
}
Write-Host "`nDone. See results\n8_A\results.json, context_horizon.png, rmse_vs_k.png" -ForegroundColor Green
