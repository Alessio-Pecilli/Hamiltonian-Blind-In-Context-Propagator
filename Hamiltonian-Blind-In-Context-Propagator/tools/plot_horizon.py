"""Summary figure: how far ahead the forecast stays good, against the context length.

    python -m tools.plot_horizon --export ../docs/data/runs/<export>.json --out ../docs/figures/horizon_vs_context.png

Reads a dashboard export (tools/export_dashboard.py). Left: last step k up to which the correlation between
predicted and true single-qubit observables stays above --thr, for every context T (the dashed diagonal is k = T).
Right: that correlation against k for the shortest context."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def reach(curve, thr):
    """Last k with curve(k') >= thr for all k' <= k, and whether the curve never dropped (tested limit hit)."""
    bad = np.nonzero(np.asarray(curve) < thr)[0]
    return (int(bad[0]), False) if len(bad) else (len(curve), True)


def find(results, name, method):
    """Method curves by T from the result set `name`, or from the first set that has the method."""
    sets = [r for r in results if r["name"] == name] + results
    return next(r["methods"][method] for r in sets if method in r["methods"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--model_set", default="roll out alternato, blocchi da 16")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out", default="horizon_vs_context.png")
    a = ap.parse_args()
    res = json.load(open(a.export))["results"]
    series = [("Transformer", BLUE, "o", find(res, a.model_set, "model(A)")), ("DMD", ORANGE, "s", find(res, a.model_set, "dmd"))]
    Ts = sorted(int(t) for t in series[0][3])

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURF, gridspec_kw=dict(wspace=0.28))
    for x in (ax, bx):
        x.set_facecolor(SURF); x.grid(color=GRID, lw=0.8); x.set_axisbelow(True)

    # left: reach against context
    Hmax = max(len(series[0][3][str(T)]["corr_n"]) for T in Ts)
    ax.plot([Ts[0] / 1.3, Ts[-1] * 1.3], [Ts[0] / 1.3, Ts[-1] * 1.3], ls="--", color=MUTED, lw=1)
    ax.text(90, 100, "k = T", color=MUTED, rotation=38, ha="center", va="bottom", fontsize=9)
    ax.axhline(Hmax, color=MUTED, lw=1, ls=":")
    ax.text(Ts[0] / 1.3, Hmax * 1.45, f"limite testato: {Hmax} passi", color=MUTED, fontsize=9, va="bottom")
    for name, col, mk, byT in series:
        pts = [(T, *reach(byT[str(T)]["corr_n"], a.thr), len(byT[str(T)]["corr_n"])) for T in Ts]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, zorder=3)
        for T, k, capped, H in pts:  # hollow marker: the correlation never dropped within the tested steps
            ax.plot(T, k, mk, ms=8, color=col, mfc=SURF if capped else col, mew=2, zorder=4)
            ax.annotate(f"{'≥ ' if capped else ''}{k}", (T, k), xytext=(0, 9 if (name == "DMD") == (T == Ts[-1]) else -16),
                        textcoords="offset points", ha="center", color=MUTED, fontsize=8.5)
        ax.text(pts[2][0], pts[2][1] * (0.62 if name == "Transformer" else 1.5), name, color=INK, fontsize=10, fontweight="bold", ha="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.set_yticks([2, 4, 8, 16, 32, 64]); ax.set_yticklabels([2, 4, 8, 16, 32, 64])
    ax.set_xlim(Ts[0] / 1.3, Ts[-1] * 1.3); ax.set_ylim(2, 150)
    ax.set_xlabel("contesto T (passi osservati)"); ax.set_ylabel(f"orizzonte k con correlazione ≥ {a.thr:g}")
    ax.set_title("Fin dove regge la previsione", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.plot([], [], "o", ms=8, color=MUTED, mfc=SURF, mew=2, label="mai sotto soglia nei passi testati")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=MUTED)

    # right: correlation against k for the shortest context
    T0 = Ts[0]
    for name, col, mk, byT in series:
        c = byT[str(T0)]["corr_n"]
        k = np.arange(1, len(c) + 1)
        bx.plot(k / T0, c, color=col, lw=2)
        bx.text(k[-1] / T0 * 1.05, c[-1], name, color=INK, fontsize=10, fontweight="bold", va="center")
    bx.axvline(1, color=MUTED, lw=1, ls="--"); bx.text(1.06, 0.04, "k = T", color=MUTED, fontsize=9)
    bx.axhline(a.thr, color=MUTED, lw=1, ls=":")
    bx.set_xscale("log", base=2); bx.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); bx.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    bx.set_xlim(0.125, 16); bx.set_ylim(0, 1.02)
    bx.set_xlabel("orizzonte / contesto  (k / T)"); bx.set_ylabel("correlazione previsto–vero")
    bx.set_title(f"Contesto più corto: T = {T0} passi", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor=SURF)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
