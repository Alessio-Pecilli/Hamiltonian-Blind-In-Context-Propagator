"""Baselines (Sec. 7.1), kept deliberately simple:
  persistence, in-context ridge DMD (w=1 and w=4), oracle, Hamiltonian learning + oracle state.

All functions work on one trajectory window: context node (T+1, n, 3), edge (T+1, n, n, 9),
and return predictions node (H, n, 3), edge (H, n, n, 9) for steps T+1..T+H.
"""
from __future__ import annotations

import time

import h5py
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from data import physics
from data.generate import simulate
from eval.metrics import k_star, rmse_per_traj

SWAP9 = np.array([3 * b + a for a in range(3) for b in range(3)])


# ----------------------------------------------------------------------------------------
# flatten / unflatten the M = 3n + 9n(n-1)/2 independent numbers of o_t
# ----------------------------------------------------------------------------------------
def flatten_obs(node: np.ndarray, edge: np.ndarray) -> np.ndarray:
    n = node.shape[-2]
    iu, ju = np.triu_indices(n, 1)
    return np.concatenate([node.reshape(*node.shape[:-2], -1),
                           edge[..., iu, ju, :].reshape(*edge.shape[:-3], -1)], -1)


def unflatten_obs(x: np.ndarray, n: int):
    lead = x.shape[:-1]
    node = x[..., : 3 * n].reshape(*lead, n, 3)
    iu, ju = np.triu_indices(n, 1)
    up = x[..., 3 * n:].reshape(*lead, len(iu), 9)
    edge = np.zeros((*lead, n, n, 9), x.dtype)
    edge[..., iu, ju, :] = up
    edge[..., ju, iu, :] = up[..., SWAP9]
    return node, edge


# ----------------------------------------------------------------------------------------
# 1. persistence
# ----------------------------------------------------------------------------------------
def persistence(node, edge, H, **_):
    return np.repeat(node[-1:], H, 0), np.repeat(edge[-1:], H, 0)


# ----------------------------------------------------------------------------------------
# 2. in-context ridge DMD / DLinear
# ----------------------------------------------------------------------------------------
def ridge_dmd(node, edge, H, w: int = 1, lam: float = 1e-3, **_):
    """Fit o_{t+1} = W [o_t; ...; o_{t-w+1}] by ridge on the context, roll out W^k.
    Uses the dual (T x T) form, so cost does not depend on M^2."""
    n = node.shape[1]
    o = flatten_obs(node, edge).astype(np.float64)        # (T+1, M)
    Tp1, M = o.shape
    if Tp1 - w < 1:
        return persistence(node, edge, H)
    X = np.stack([np.concatenate([o[t - s] for s in range(w)]) for t in range(w - 1, Tp1 - 1)])
    Y = o[w:]                                               # (P, M)
    G = X @ X.T + lam * np.eye(len(X))
    Wt = X.T @ np.linalg.solve(G, Y)                        # (wM, M):  o_{t+1} = feat @ Wt
    hist = [o[-1 - s] for s in range(w)]
    preds = []
    for _ in range(H):
        nxt = np.concatenate(hist) @ Wt
        preds.append(nxt)
        hist = [nxt] + hist[:-1]
    p = np.clip(np.stack(preds), -1, 1)                     # observables are bounded
    pn, pe = unflatten_obs(p, n)
    return pn.astype(np.float32), pe.astype(np.float32)


# ----------------------------------------------------------------------------------------
# 3. oracle and 4. HL + oracle state
# ----------------------------------------------------------------------------------------
def _propagate(Hmat, psi, dt, H):
    n = int(np.log2(psi.shape[0]))
    Psi = physics.evolve(Hmat, psi, dt, H + 1)[1:]
    return physics.observables(Psi, n)


def oracle_state(attrs, seed, t_abs):
    """True |psi_t> regenerated from the stored seed."""
    tr = simulate(int(attrs["n"]), t_abs + 1, float(attrs["dt"]), str(attrs["family"]), int(seed),
                  keep_state=True, check=False)
    return tr.Psi[t_abs], tr.H


def oracle(node, edge, H, attrs=None, seed=None, t_abs=None, **_):
    psi, Hmat = oracle_state(attrs, seed, t_abs)
    pn, pe = _propagate(Hmat, psi, float(attrs["dt"]), H)
    return pn.astype(np.float32), pe.astype(np.float32)


def lasso_ista(A, b, alpha, iters=3000, tol=1e-10):
    """min 1/(2N) ||Ax - b||^2 + alpha ||x||_1   (FISTA)."""
    N = A.shape[0]
    Lc = np.linalg.norm(A, 2) ** 2 / N
    x = np.zeros(A.shape[1]); y = x.copy(); tk = 1.0
    for _ in range(iters):
        g = A.T @ (A @ y - b) / N
        z = y - g / Lc
        x_new = np.sign(z) * np.maximum(np.abs(z) - alpha / Lc, 0.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        y = x_new + (tk - 1) / t_new * (x_new - x)
        if np.abs(x_new - x).max() < tol:
            x = x_new
            break
        x, tk = x_new, t_new
    return x


def estimate_tfim(node, edge, dt, l1=1e-3):
    """Least squares on the single-site Heisenberg equations of the TFIM (Eq. 29), all pairs
    allowed. Unknowns: h (n) and J_ij for i<j. Returns J (n,n), h (n)."""
    node, edge = node.astype(np.float64), edge.astype(np.float64)
    Tp1, n, _ = node.shape
    iu, ju = np.triu_indices(n, 1)
    P = len(iu)
    pair = -np.ones((n, n), int)
    pair[iu, ju] = np.arange(P); pair[ju, iu] = np.arange(P)
    d = (node[2:] - node[:-2]) / (2 * dt)                     # (T-1, n, 3)
    Nm, Em = node[1:-1], edge[1:-1]
    rows, rhs = [], []
    X, Y, Z = 0, 1, 2
    for t in range(Tp1 - 2):
        for i in range(n):
            rx, ry, rz = np.zeros(n + P), np.zeros(n + P), np.zeros(n + P)
            for j in range(n):
                if j == i:
                    continue
                p = n + pair[i, j]
                rx[p] = 2 * Em[t, i, j, 3 * Y + Z]               # dX = 2 sum J <Y_i Z_j>
                ry[p] = -2 * Em[t, i, j, 3 * X + Z]              # dY = -2 sum J <X_i Z_j> + 2h<Z>
            ry[i] = 2 * Nm[t, i, Z]
            rz[i] = -2 * Nm[t, i, Y]                             # dZ = -2 h <Y>
            rows += [rx, ry, rz]
            rhs += [d[t, i, X], d[t, i, Y], d[t, i, Z]]
    x = lasso_ista(np.array(rows), np.array(rhs), l1)
    h = x[:n]
    J = np.zeros((n, n)); J[iu, ju] = x[n:]; J = J + J.T
    return J, h


def hl_oracle(node, edge, H, attrs=None, seed=None, t_abs=None, **_):
    dt = float(attrs["dt"])
    J, h = estimate_tfim(node, edge, dt)
    psi, _ = oracle_state(attrs, seed, t_abs)
    n = node.shape[1]
    Hmat = physics.build_hamiltonian(n, physics.tfim_terms(J, h))
    pn, pe = _propagate(Hmat, psi, dt, H)
    return pn.astype(np.float32), pe.astype(np.float32)


BASELINES = dict(
    persistence=persistence,
    dmd=lambda *a, **k: ridge_dmd(*a, w=1, **k),
    dmd4=lambda *a, **k: ridge_dmd(*a, w=4, **k),
    oracle=oracle,
    hl_oracle=hl_oracle,
)


def run_baselines(path, names, T_list, horizon, max_traj=None, t0=0, verbose=True) -> dict:
    """Same output format as eval.metrics.evaluate_model: {name: {T: result}}."""
    out = {}
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        L = f["node"].shape[1]
        N = f["node"].shape[0] if max_traj is None else min(max_traj, f["node"].shape[0])
        node_all, edge_all, seeds = f["node"][:N], f["edge"][:N], f["seed"][:N]
    if "hl_oracle" in names and attrs["family"] != "tfim":
        raise ValueError("hl_oracle baseline only implements the TFIM estimator")
    for name in names:
        fn = BASELINES[name]
        out[name] = {}
        for T in T_list:
            Hk = min(horizon, L - 1 - T - t0)
            if Hk < 1:
                continue
            rn, re, t_tot = [], [], 0.0
            for k in range(N):
                ctx_n, ctx_e = node_all[k, t0: t0 + T + 1], edge_all[k, t0: t0 + T + 1]
                tgt_n, tgt_e = node_all[k, t0 + T + 1: t0 + T + 1 + Hk], edge_all[k, t0 + T + 1: t0 + T + 1 + Hk]
                ts = time.perf_counter()
                pn, pe = fn(ctx_n, ctx_e, Hk, attrs=attrs, seed=seeds[k], t_abs=t0 + T)
                t_tot += time.perf_counter() - ts
                a, b = rmse_per_traj(pn[None], tgt_n[None], pe[None], tgt_e[None])
                rn.append(a[0]); re.append(b[0])
            cn, ce = np.mean(rn, 0), np.mean(re, 0)
            out[name][T] = dict(T=T, H=Hk, ntraj=N, rmse_n=cn.tolist(), rmse_e=ce.tolist(),
                                k_star=k_star(cn), sec_per_step=t_tot / (N * Hk))
            if verbose:
                print(f"  {name:12s} T={T:4d}  rmse_n(1)={cn[0]:.4f}  rmse_n(H)={cn[-1]:.4f}  "
                      f"k*={out[name][T]['k_star']}", flush=True)
    return out
