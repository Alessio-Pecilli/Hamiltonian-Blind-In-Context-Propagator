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
| `eval/metrics.py` | 6 | RMSE_n,e(k), k*(T), graph AUROC (gate and spatial attention), probe R²/AUROC, ‖S‖_F, cost |
| `eval/baselines.py` | 7.1 | persistence, ridge DMD (w = 1, 4), oracle, HL + oracle state |
| `eval/evaluate.py` | 6–7 | evaluation CLI, `results.json`, context–horizon and RMSE figures |
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

Any config key can be overridden with `--set section.key=value`.
Large datasets: keep `data.preload=false` (HDF5 is read lazily per item).

## Conventions and choices not fixed by the spec

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
