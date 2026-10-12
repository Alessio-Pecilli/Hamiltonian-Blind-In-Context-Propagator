# Hamiltonian-blind in-context propagator

PyTorch implementation of `propagator_spec.pdf` (v0.1): a space–time transformer over site tokens
and pair features with an in-context structured generator head, forecasting the weight-≤2 Pauli
observables of an n-qubit system with unknown Hamiltonian and unknown interaction graph.

Not implemented in v0.1 (on purpose): shot noise (Sec. 2.5), noise / Lindblad stage (Sec. 5.3.4).
Baselines (Sec. 7.1) are deliberately lightweight.

## Layout

| file | spec | content |
|---|---|---|
| `data/physics.py` | 2.3–2.4 | sparse Pauli Hamiltonians, exact evolution (dense `expm` for n ≤ 12, Krylov above), observables |
| `data/generate.py` | 2.1–2.2, 2.6 | samplers (`tfim`, plus `xxz`, `heisenberg`, `lr_ising`), HDF5 writer, CLI |
| `data/dataset.py` | 2.7, 5.1 | windowed loader: T, t0, anchors, site permutation, right-padding collate |
| `model/embed.py`, `spatial.py`, `temporal.py`, `body.py` | 3.2–3.5 | embeddings, pair-biased spatial attention + pair update, causal MHA (RoPE) + causal depthwise conv |
| `model/head_generator.py` | 3.6 | Head A: G = (A − Aᵀ) − SSᵀ, `matrix_exp`, linear/bilinear decoders |
| `model/head_free.py`, `model/probe.py` | 3.7–3.8 | Head B (ablation), stop-gradient probe |
| `model/propagator.py` | 3.1 | full model |
| `losses.py` | 4 | L_pred, L_phys (ρ̂ ≥ 0 via `eigvalsh`), L_probe, ‖θ‖² |
| `train.py` | 5 | AdamW, warm-up + cosine, clipping, bf16 body / fp32 head, checkpoints (config + git commit), TensorBoard + `log.jsonl` |
| `eval/metrics.py` | 6 | RMSE_n,e(k), k*(T), reduced two-qubit state fidelity F₂(k), graph AUROC (gate and spatial attention), probe R²/AUROC, ‖S‖_F, cost |
| `eval/baselines.py` | 7.1 | persistence, ridge DMD (w = 1, 4), oracle, HL + oracle state |
| `eval/evaluate.py` | 6–7 | evaluation CLI, `results.json`, context–horizon and RMSE figures |
| `tools/export_dashboard.py` | – | collects runs, results, datasets and a sample rollout into one JSON (`dashboard_data.json`) |
| `tools/publish_dashboard.py` | – | export → `docs/data/runs/…json` → `index.json` → git commit + push (GitHub Pages) |
| `../docs/` | – | the dashboard (static site served by GitHub Pages): `index.html`, `style.css`, `app.js`, `glossary.js`, `model-info.js`, `data/` |
| `configs/config.yaml` | 3.9 | single config with the `local` and `hpc` presets |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # all unit tests of the implementation plan (Sec. 8)

bash scripts/make_datasets.sh local 4          # n = 8: 1000/100/100 trajectories
python train.py --preset local --overfit       # Stage 1 check (i): single-batch overfit to L_pred < 1e-4
python train.py --preset local                 # Stage 1 training (~20k steps)
python -m eval.evaluate --ckpt runs/default/last.pt --data datasets/n8/test.h5 \
    --T 8 16 32 64 --horizon 64 --baselines persistence dmd dmd4 oracle hl_oracle --out results/n8
```

HPC (Stage 2, single GPU):

```bash
bash scripts/make_datasets.sh hpc 32
python train.py --preset hpc --set train.out_dir=runs/hpc_A
python train.py --preset hpc --set train.out_dir=runs/hpc_B model.head=B     # ablation
bash scripts/make_datasets.sh transfer 32                                   # Stage 3 test sets
python -m eval.evaluate --ckpt runs/hpc_A/last.pt --data datasets/n12/test.h5 --out results/n12
```

## Dashboard

The dashboard (results, input/output on one trajectory, inferred graph, equations, loss terms, training
curves and a glossary with a popup on every quantity) is a static site in `docs/` at the root of the
git repository, next to this folder. Live version:

**https://alessio-pecilli.github.io/Hamiltonian-Blind-In-Context-Propagator/**

Enable it once: GitHub → *Settings → Pages → Build and deployment → Deploy from a branch* →
branch `main`, folder `/docs`. Pages needs a **public repository** or a GitHub Pro (or higher) plan
for a private one. `docs/.nojekyll` is already there.

Publish a new version after training and evaluation (from this folder, PowerShell or bash):

```bash
python -m tools.publish_dashboard --label "n8 no S" --notes "use_S=false"     --runs runs --results results --data datasets/n8/test.h5     --ckpt runs/local_A/last.pt --T 32 --horizon 64 --sample 0
```

It writes `docs/data/runs/<YYYYmmdd-HHMMSS>_<label>.json`, puts it first (and as `latest`) in
`docs/data/index.json`, keeps the last `--keep N` exports (default 20), then runs `git add docs/data`,
`git commit -m "dashboard: <label>"` and `git push`. Options: `--no-push`, `--dry-run`, `--keep`; it warns
when the JSON exceeds 5 MB. Pages updates about a minute after the push. On Windows,
`scriptsun_all.ps1 -Publish` does the whole pipeline and publishes at the end.

The page opens `data/index.json`, loads `latest`, and lets you pick older exports in the *Export* menu.
Without the manifest it falls back to `data/demo.json` (a small cloud demo run). A local `dashboard_data.json`
(from `python -m tools.export_dashboard --out dashboard_data.json`) can be opened with the *Carica* button.
To look at it locally: `python -m http.server` inside `docs/`, then open http://localhost:8000
(opening `index.html` from disk blocks the JSON requests). `?dev` in the URL logs glossary terms without an entry.
`runs/`, `results/`, `datasets/`, `*.pt` and `*.h5` stay out of git.

Any config key can be overridden with `--set section.key=value`.
Large datasets: keep `data.preload=false` (HDF5 is read lazily per item).

## Conventions and choices not fixed by the spec

* `model.use_S: false` (default in `configs/config.yaml`): no dissipation, `G = A − Aᵀ` is skew-symmetric and `‖S‖_F = 0`;
  configs without the key keep the old behaviour (S active), so older checkpoints still load.
* Pair-path options (`configs/config.yaml`, all on by default; configs without the keys keep the old network, so older
  checkpoints still load; ablate with `--set model.pair_attn=false` etc.):
  `model.pair_attn` causal temporal attention (RoPE) on every pair, besides the depthwise conv;
  `model.triangle` triangular update `e_ij += Tri((1/n) Σ_k a_ik ⊙ a_jk)` (symmetric, gated);
  `model.pair_latent = p` per-pair latent `u_ij`, initialised from `ē_ij` and rotated by its own skew generator
  `C_ij(ē_ij)`, added to the pair decoder (`D_u` and `W_C` are zero-initialised: the model starts as before).
* Fidelity F₂(k): mean over pairs i<j of the Uhlmann fidelity between the two-qubit marginal rebuilt from the prediction
  (Eq. 28, projected on the PSD cone) and the one from the true observables. The full-state fidelity is not
  defined for a model that only outputs weight-≤2 observables.
* Site 0 is the most significant qubit; edge channel `c = 3a + b`, `a, b ∈ {x, y, z}`.
* `e_ii = 0` is kept throughout the body (diagonal masked after every update), not only at embedding.
* Value augmentation is computed as `W_ev (Σ_j α_ij LN(e_ij))` (identical by linearity, O(n² d_e) memory).
* Bilinear decoder bases `B_c` are initialised N(0, 1/m²); `β_c = 0`.
* `||S||_F` is reported for the full block-diagonal S (= √n · ‖S_block‖_F).
* Time-averaged attention for graph recovery = mean over blocks, heads and context positions, symmetrised.
* DMD predictions are clipped to [−1, 1]; the ridge is solved in its dual (T×T) form.
* HL baseline: FISTA lasso, objective `1/(2N)‖Ax − b‖² + 10⁻³‖x‖₁`; TFIM family only.
* Spec inconsistency: with `L_traj = 160`, `T = 128` and `H = 32` do not fit (`t0 + T + H < L_traj`);
  the HPC datasets are generated with `L_traj = 161` (train/val) and `192` (test, k up to 64).
* Parameter counts with the spec widths: ≈1.46M (local) and ≈11.2M (HPC), vs. the ≈1.2M / ≈9M estimates.
