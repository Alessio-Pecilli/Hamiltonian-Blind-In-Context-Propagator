"""Why DMD looks better with a long context: context length against point of the trajectory.

    python -m tools.plot_dmd_context --data datasets/n10/test.h5 --out ../docs/figures/dmd_context_vs_time.png

Windows that start at t = 0 (the standard evaluation) forecast later and later as T grows, where the signal is
smaller. The same DMD is run with every context ending at the same step (--start), so only T changes."""
from __future__ import annotations

import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.baselines import run_baselines
from tools.plot_horizon import BLUE, GRID, INK, MUTED, ORANGE, SURF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--T", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--start", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_traj", type=int, default=48)
    ap.add_argument("--out", default="dmd_context_vs_time.png")
    a = ap.parse_args()
    with h5py.File(a.data, "r") as f:
        node = f["node"][: a.max_traj]
    amp = np.sqrt((node ** 2).mean((2, 3))).mean(0)                       # (L,) rms of the single-qubit observables
    run = lambda T, t0: run_baselines(a.data, ["dmd"], [T], a.k, a.max_traj, t0=t0, verbose=False)["dmd"][T]
    zero = [run(T, 0) for T in a.T]
    fixed = [run(T, a.start - T) for T in a.T]
    for T, z, x in zip(a.T, zero, fixed):
        print(f"T={T:4d}  from t=0: rmse {z['rmse_n'][-1]:.3f} rel {z['rel_rmse_n'][-1]:.2f}   "
              f"ending at {a.start}: rmse {x['rmse_n'][-1]:.3f} rel {x['rel_rmse_n'][-1]:.2f}")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURF, gridspec_kw=dict(wspace=0.3))
    for x in axs:
        x.set_facecolor(SURF); x.grid(color=GRID, lw=0.8); x.set_axisbelow(True)
    ax, bx, cx = axs
    ax.plot(np.arange(len(amp)), amp, color=INK, lw=2)
    for T in a.T:  # where a window that starts at 0 begins to forecast
        ax.plot(T, amp[T], "o", ms=8, color=ORANGE, mec=SURF, mew=2, zorder=3)
        ax.annotate(f"T = {T}", (T, amp[T]), xytext=(6, 6), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.set_xlabel("passo temporale"); ax.set_ylabel("ampiezza del segnale (osservabili a un qubit)")
    ax.set_ylim(0, None); ax.set_xlim(0, len(amp))
    ax.set_title("Il segnale si spegne nel tempo", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.text(0.97, 0.9, "punti: da dove parte la previsione\nse la finestra comincia a t = 0", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=8.5)
    for x, key, ylab, title in ((bx, "rmse_n", f"RMSE del DMD a k = {a.k}", "Errore assoluto"),
                                (cx, "rel_rmse_n", f"errore / ampiezza del segnale a k = {a.k}", "Errore relativo")):
        for res, col, mk, name, dy in ((zero, ORANGE, "s", "finestra da t = 0", 10), (fixed, BLUE, "o", f"previsione sempre dal passo {a.start}", -16)):
            y = [r[key][-1] for r in res]
            x.plot(a.T, y, color=col, lw=2); x.plot(a.T, y, mk, ms=8, color=col, mec=SURF, mew=2, label=name)
        x.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
        x.set_xscale("log", base=2); x.set_xticks(a.T); x.set_xticklabels(a.T); x.set_ylim(0, None)
        x.set_xlabel("contesto T del DMD"); x.set_ylabel(ylab)
        x.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor=SURF)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
