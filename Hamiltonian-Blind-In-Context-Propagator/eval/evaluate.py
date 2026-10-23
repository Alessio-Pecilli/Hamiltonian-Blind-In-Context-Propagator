"""Evaluation on a test split (Sec. 6-7): model + baselines, JSON output and the two figures.

    python -m eval.evaluate --ckpt runs/default/last.pt --data datasets/n8/test.h5 \
        --T 8 16 32 64 --horizon 64 --baselines persistence dmd dmd4 oracle hl_oracle
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from eval.baselines import run_baselines
from eval.metrics import evaluate_model
from model.propagator import Propagator


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Propagator(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def plot(results: dict, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping figures")
        return
    # context-horizon curve k*(T)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, res in results.items():
        Ts = sorted(int(t) for t in res)
        ax.plot(Ts, [res[t]["k_star"] for t in Ts], "o-", label=name)
    ax.set_xscale("log", base=2); ax.set_xlabel("context length T"); ax.set_ylabel("k* (eps=0.05)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "context_horizon.png"), dpi=150)
    # RMSE_n(k) at each T
    Ts = sorted({int(t) for r in results.values() for t in r})
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.2 * len(Ts), 3), sharey=True, squeeze=False)
    for ax, T in zip(axes[0], Ts):
        for name, res in results.items():
            if T in res:
                ax.plot(np.arange(1, len(res[T]["rmse_n"]) + 1), res[T]["rmse_n"], label=name)
        ax.axhline(0.05, color="k", lw=0.5, ls="--"); ax.set_title(f"T={T}"); ax.set_xlabel("k")
        ax.set_yscale("log")
    axes[0, 0].set_ylabel("RMSE_n(k)"); axes[0, -1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "rmse_vs_k.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--T", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--baselines", nargs="*", default=["persistence", "dmd", "dmd4"],
                    help="any of persistence dmd dmd4 oracle hl_oracle hl_fair_o0 hl_fair_fit")
    ap.add_argument("--max_traj", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device)
    results = {}
    if a.ckpt:
        model, ck = load_model(a.ckpt, device)
        mb = None if a.max_traj is None else -(-a.max_traj // a.batch_size)
        res = evaluate_model(model, a.data, a.T, a.horizon, device, a.batch_size, max_batches=mb)
        results[f"model({model.head_type})"] = res
        for T, r in res.items():
            print(f"  model        T={T:4d}  rmse_n(1)={r['rmse_n'][0]:.4f}  rmse_n(H)={r['rmse_n'][-1]:.4f}  "
                  f"k*={r['k_star']}  AUROC gate={r.get('auroc_gate', float('nan')):.3f} "
                  f"attn={r.get('auroc_attn', float('nan')):.3f}  S_F={r.get('S_F', float('nan')):.3g}  "
                  f"G: auroc={r.get('gen_auroc_adj', float('nan')):.3f} corrJ={r.get('gen_corr_J', float('nan')):.3f} "
                  f"corrh={r.get('gen_corr_h', float('nan')):.3f}")
    if a.baselines:
        results.update(run_baselines(a.data, a.baselines, a.T, a.horizon, a.max_traj))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({k: {str(t): r for t, r in v.items()} for k, v in results.items()}, f, indent=1)
    plot(results, a.out)
    print(f"wrote {a.out}/results.json")


if __name__ == "__main__":
    main()
