"""Summary figure: how far ahead the forecast stays good, against the context length, on the RELATIVE error
(RMSE of the single-qubit observables divided by the RMS of the true signal at that step).

    python -m tools.plot_horizon --export ../docs/data/runs/<export>.json --out ../docs/figures/horizon_vs_context.png

Reads a dashboard export (tools/export_dashboard.py). Left: k*_rel, the last step k up to which the relative error stays
below each threshold, for every context T (dashed diagonal: k = T). Middle and right: the relative error against k for
the shortest and the longest context."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def reach(curve, thr):
    """k*_rel: last k with curve(k') <= thr for all k' <= k, and whether it never crossed (tested limit hit)."""
    bad = np.nonzero(np.asarray(curve) > thr)[0]
    return (int(bad[0]), False) if len(bad) else (len(curve), True)


def find(results, name, method):
    """Method curves by T from the result set `name`, or from the first set that has the method."""
    sets = [r for r in results if r["name"] == name] + results
    return next(r["methods"][method] for r in sets if method in r["methods"])


def style(x):
    x.set_facecolor(SURF); x.grid(color=GRID, lw=0.8); x.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--model_set", default="roll out alternato, blocchi da 16")
    ap.add_argument("--thr", type=float, nargs=2, default=[0.5, 0.25], help="main and secondary threshold")
    ap.add_argument("--T_train_max", type=int, default=96, help="longest context seen in training (marked)")
    ap.add_argument("--out", default="horizon_vs_context.png")
    a = ap.parse_args()
    res = json.load(open(a.export))["results"]
    series = [("Transformer", BLUE, "o", find(res, a.model_set, "model(A)")), ("DMD", ORANGE, "s", find(res, a.model_set, "dmd"))]
    Ts = sorted(int(t) for t in series[0][3])
    t1, t2 = a.thr

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=SURF, gridspec_kw=dict(wspace=0.3, width_ratios=[1.15, 1, 1]))
    for x in axs:
        style(x)
    ax = axs[0]
    Hmax = max(len(series[0][3][str(T)]["rel_rmse_n"]) for T in Ts)
    ax.plot([Ts[0] / 1.3, Ts[-1] * 1.3], [Ts[0] / 1.3, Ts[-1] * 1.3], ls="--", color=MUTED, lw=1)
    ax.text(13, 8.6, "k = T", color=MUTED, rotation=33, ha="center", va="bottom", fontsize=9)
    capped_any = any(reach(byT[str(T)]["rel_rmse_n"], t1)[1] for *_, byT in series for T in Ts)
    if capped_any:
        ax.axhline(Hmax, color=MUTED, lw=1, ls=":")
        ax.text(Ts[-1] * 1.25, Hmax * 1.1, f"limite testato: {Hmax} passi", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    ax.axvspan(a.T_train_max, Ts[-1] * 1.3, color=GRID, alpha=0.6, lw=0)
    ax.text(Ts[-1] * 1.25, 1.15, "contesto oltre\nil training", color=MUTED, fontsize=8, ha="right", va="bottom")
    for name, col, mk, byT in series:
        for thr, lw, ls, ms in ((t1, 2, "-", 8), (t2, 1.2, (0, (3, 2)), 5)):
            pts = [(T, *reach(byT[str(T)]["rel_rmse_n"], thr)) for T in Ts]
            ax.plot([p[0] for p in pts], [max(p[1], 1) for p in pts], color=col, lw=lw, ls=ls, zorder=3)
            for T, k, capped in pts:
                ax.plot(T, max(k, 1), mk, ms=ms, color=col, mfc=SURF if capped else col, mew=1.6, zorder=4)
                if thr == t1:
                    ax.annotate(f"{'≥ ' if capped else ''}{k}", (T, k), xytext=(0, 9 if name == "Transformer" or T >= 64 else -15),
                                textcoords="offset points", ha="center", color=MUTED, fontsize=8.5)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.set_yticks([1, 2, 4, 8, 16, 32, 64]); ax.set_yticklabels([1, 2, 4, 8, 16, 32, 64])
    ax.set_xlim(Ts[0] / 1.3, Ts[-1] * 1.3); ax.set_ylim(1, 400)
    ax.set_xlabel("contesto T (passi osservati)"); ax.set_ylabel("k*_rel: orizzonte con errore relativo sotto soglia")
    ax.set_title("Fin dove regge la previsione", loc="left", color=INK, fontsize=11, fontweight="bold")
    h = [plt.Line2D([], [], color=c, marker=m, lw=2, ms=7) for _, c, m, _ in series]
    h += [plt.Line2D([], [], color=MUTED, lw=2), plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 2))),
          plt.Line2D([], [], color=MUTED, marker="o", mfc=SURF, lw=0, ms=7, mew=1.6)]
    lab = ["Transformer", "DMD", f"soglia {t1:.0%}", f"soglia {t2:.0%}", "mai sopra soglia nei passi testati"]
    n_leg = 5 if capped_any else 4
    ax.legend(h[:n_leg], lab[:n_leg], loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK, ncol=2)

    for x, T in ((axs[1], Ts[0]), (axs[2], Ts[-1])):
        for name, col, mk, byT in series:
            c = np.asarray(byT[str(T)]["rel_rmse_n"])
            k = np.arange(1, len(c) + 1)
            x.plot(k, c, color=col, lw=2, label=name)
        for thr, ls in ((t1, ":"), (t2, ":")):
            x.axhline(thr, color=MUTED, lw=1, ls=ls)
            x.text(k[-1] if T == Ts[0] else 1.05, thr + 0.02, f"{thr:.0%}", color=MUTED, fontsize=8.5,
                   ha="right" if T == Ts[0] else "left")
        x.set_xscale("log", base=2); x.set_xticks([1, 2, 4, 8, 16, 32, 64]); x.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        x.set_ylim(0, 1.05); x.set_xlabel("passi in avanti k"); x.set_ylabel("errore / ampiezza del segnale")
        out = f"  (oltre il training: max {a.T_train_max})" if T > a.T_train_max else ""
        x.set_title(f"Contesto T = {T} passi{out}", loc="left", color=INK, fontsize=11, fontweight="bold")
        x.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor=SURF)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
