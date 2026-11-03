"""Metrics (Sec. 6): horizon RMSE, predictability horizon k*, graph recovery, probe quality,
dissipation gauge, cost."""
from __future__ import annotations

import time

import h5py
import numpy as np
import torch

from data.dataset import gather_targets, make_loader

EPS_KSTAR = 0.05
EPS_REL = 0.25  # relative k*: error below 25% of the true signal amplitude at that step


# ----------------------------------------------------------------------------------------
# elementary metrics
# ----------------------------------------------------------------------------------------
def rmse_per_traj(pn, tn, pe, te):
    """Inputs (N, H, n, 3) and (N, H, n, n, 9) -> per-trajectory RMSE_n, RMSE_e of shape (N, H)."""
    n = pn.shape[-2]
    rn = ((pn - tn) ** 2).mean(axis=(-1, -2)) ** 0.5
    mask = ~np.eye(n, dtype=bool)
    re = (((pe - te) ** 2)[..., mask, :]).mean(axis=(-1, -2)) ** 0.5
    return rn, re


class RelMetrics:
    """Streaming relative metrics per horizon step k, nodes and pairs separately: RMSE divided by
    the RMS of the true signal (both averaged over trajectories), and the Pearson correlation of
    predicted vs true values over all trajectories and entries. Scale-free: the observables shrink
    as the state spreads, so a fixed absolute threshold favours late windows."""

    def __init__(self):
        self.s = {}

    @staticmethod
    def _flat(pn, tn, pe, te):
        n = pn.shape[-2]
        mask = ~np.eye(n, dtype=bool)
        N, H = pn.shape[:2]
        return dict(n=(pn.reshape(N, H, -1), tn.reshape(N, H, -1)),
                    e=(pe[:, :, mask].reshape(N, H, -1), te[:, :, mask].reshape(N, H, -1)))

    def add(self, pn, tn, pe, te):
        for key, (p, t) in self._flat(pn, tn, pe, te).items():
            p, t = p.astype(np.float64), t.astype(np.float64)
            acc = dict(amp=(t ** 2).mean(-1) ** 0.5, x=p, y=t, xx=p * p, yy=t * t, xy=p * t)
            acc = {k: v.sum(axis=0) if k == "amp" else v.sum(axis=(0, 2)) for k, v in acc.items()}
            acc["c"] = np.full(p.shape[1], p.shape[0] * p.shape[2], dtype=np.float64)
            acc["N"] = p.shape[0]
            old = self.s.get(key)
            self.s[key] = acc if old is None else {k: old[k] + v for k, v in acc.items()}

    def result(self, rmse_n, rmse_e) -> dict:
        out = {}
        for key, rmse in (("n", rmse_n), ("e", rmse_e)):
            a = self.s[key]
            amp = a["amp"] / a["N"]
            c = a["c"]
            cov = a["xy"] - a["x"] * a["y"] / c
            var = (a["xx"] - a["x"] ** 2 / c) * (a["yy"] - a["y"] ** 2 / c)
            out[f"amp_{key}"] = amp.tolist()
            out[f"rel_rmse_{key}"] = (np.asarray(rmse) / np.maximum(amp, 1e-12)).tolist()
            out[f"corr_{key}"] = (cov / np.sqrt(np.maximum(var, 1e-30))).tolist()
        out["k_star_rel"] = k_star(out["rel_rmse_n"], EPS_REL)
        out["k_star_rel_e"] = k_star(out["rel_rmse_e"], EPS_REL)  # correlators: the hard, informative part
        return out


_P16 = None


def rho_pairs_np(node: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Two-qubit reduced density matrices of all pairs i<j from observables (Eq. 28):
    node (..., n, 3), edge (..., n, n, 9) -> (..., P, 4, 4) complex128."""
    global _P16
    if _P16 is None:
        from data.physics import PAULIS, SI
        ps = (SI,) + PAULIS
        _P16 = np.stack([np.stack([np.kron(a, b) for b in ps]) for a in ps])
    n = node.shape[-2]
    iu, ju = np.triu_indices(n, 1)
    lead = node.shape[:-2]
    c = np.zeros((*lead, len(iu), 4, 4))
    c[..., 0, 0] = 1.0
    c[..., 1:, 0] = node[..., iu, :]
    c[..., 0, 1:] = node[..., ju, :]
    c[..., 1:, 1:] = edge[..., iu, ju, :].reshape(*lead, len(iu), 3, 3)
    return 0.25 * np.einsum("...ab,abxy->...xy", c, _P16)


def _psd_sqrt(rho: np.ndarray):
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    return w, (v * np.sqrt(w)[..., None, :]) @ np.conj(np.swapaxes(v, -1, -2))


def fidelity_pairs(pn, tn, pe, te) -> np.ndarray:
    """Mean over pairs i<j of the Uhlmann fidelity F(rho_hat_ij, rho_ij) between the two-qubit
    reduced state rebuilt from the prediction and from the true observables.
    Inputs (N, H, n, 3) / (N, H, n, n, 9) -> (N, H). rho_hat is projected on the PSD cone
    (negative eigenvalues clipped, trace renormalised) before the comparison."""
    rh = rho_pairs_np(np.asarray(pn, np.float64), np.asarray(pe, np.float64))
    rt = rho_pairs_np(np.asarray(tn, np.float64), np.asarray(te, np.float64))
    w_, vh = np.linalg.eigh(rh)
    rh = (vh * np.clip(w_, 0, None)[..., None, :]) @ np.conj(np.swapaxes(vh, -1, -2))
    rh = rh / np.trace(rh, axis1=-2, axis2=-1).real[..., None, None]
    _, sq = _psd_sqrt(rt)
    ev = np.linalg.eigvalsh(sq @ rh @ sq)
    return (np.sqrt(np.clip(ev, 0.0, None)).sum(-1) ** 2).clip(0.0, 1.0).mean(-1)


def k_star(curve: np.ndarray, eps: float = EPS_KSTAR) -> int:
    """max{k : RMSE(k') <= eps for all k' <= k} (0 if already k=1 fails)."""
    bad = np.nonzero(~(np.asarray(curve) <= eps))[0]
    return int(bad[0]) if len(bad) else len(curve)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = np.asarray(scores, float).ravel(), np.asarray(labels, bool).ravel()
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    ss = ((true - true.mean()) ** 2).sum()
    return float(1 - ((pred - true) ** 2).sum() / ss) if ss > 0 else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def block_norms(G: np.ndarray, n: int) -> np.ndarray:
    """(N, nm, nm) -> (N, n, n) Frobenius norms of the m x m blocks of the effective generator."""
    N, nm, _ = G.shape
    m = nm // n
    return np.sqrt((G.reshape(N, n, m, n, m) ** 2).sum((2, 4)))


def upper_pairs(x: np.ndarray) -> np.ndarray:
    """(N, n, n) -> (N, n(n-1)/2) upper-triangular entries."""
    n = x.shape[-1]
    iu, ju = np.triu_indices(n, 1)
    return x[..., iu, ju]


# ----------------------------------------------------------------------------------------
# model evaluation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, path: str, T_list, horizon: int, device, batch_size: int = 16,
                   max_batches: int | None = None, t0: int = 0, graph: bool = True,
                   num_workers: int = 0) -> dict:
    """Evaluate from the last context position, sweeping T. horizon may exceed the training H
    for Head A (generator extrapolation); it is clipped to what the stored trajectories allow."""
    model.eval()
    results = {}
    for T in T_list:
        with h5py.File(path, "r") as f:
            L = f["node"].shape[1]
        Hk = min(horizon, L - 1 - T - t0)
        if model.head_type == "B":
            Hk = min(Hk, model.horizon)
        if Hk < 1:
            continue
        loader = make_loader(path, (T, T), Hk, 1, batch_size, shuffle=False, permute=False,
                             fixed_T=T, fixed_t0=t0, num_workers=num_workers)
        rn_all, re_all, fid_all, gate, attn, adj, J, h = [], [], [], [], [], [], [], []
        rel = RelMetrics()
        pJ, ph, padj, S_F, Gn = [], [], [], [], []
        n_steps, t_model = 0, 0.0
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            tn, te = gather_targets(batch, Hk)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            pred = model(batch["node"], batch["edge"], batch["dt"], batch["anchors"], batch["T"],
                         horizon=Hk, return_attn=graph)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model += time.perf_counter() - t_start
            n_steps += Hk * batch["node"].shape[0]
            rn, re = rmse_per_traj(pred["node"][:, 0].cpu().numpy(), tn[:, 0].cpu().numpy(),
                                   pred["edge"][:, 0].cpu().numpy(), te[:, 0].cpu().numpy())
            rn_all.append(rn)
            re_all.append(re)
            fid_all.append(fidelity_pairs(pred["node"][:, 0].cpu().numpy(), tn[:, 0].cpu().numpy(),
                                          pred["edge"][:, 0].cpu().numpy(), te[:, 0].cpu().numpy()))
            rel.add(pred["node"][:, 0].cpu().numpy(), tn[:, 0].cpu().numpy(),
                    pred["edge"][:, 0].cpu().numpy(), te[:, 0].cpu().numpy())
            adj.append(batch["adj"].cpu().numpy())
            J.append(batch["J"].cpu().numpy())
            h.append(batch["h"].cpu().numpy())
            if "probe" in pred:
                pJ.append(pred["probe"]["J"].cpu().numpy())
                ph.append(pred["probe"]["h"].cpu().numpy())
                padj.append(pred["probe"]["adj_logit"].cpu().numpy())
            if "G" in pred:
                Gn.append(block_norms(pred["G"].cpu().numpy(), batch["h"].shape[-1]))
            if "gate" in pred:
                gate.append(pred["gate"][:, 0].cpu().numpy())
                S_F.append(pred["S_F"][:, 0].cpu().numpy())
            if graph and "attn" in pred:
                # time-, head- and block-averaged spatial attention, symmetrised
                a = torch.stack([x.float().mean((1, 2)) for x in pred["attn"]]).mean(0)
                attn.append((0.5 * (a + a.transpose(-1, -2))).cpu().numpy())
        rn, re = np.concatenate(rn_all), np.concatenate(re_all)
        adj, J, h = np.concatenate(adj), np.concatenate(J), np.concatenate(h)
        curve_n, curve_e = rn.mean(0), re.mean(0)
        res = dict(T=T, H=Hk, ntraj=len(rn), rmse_n=curve_n.tolist(), rmse_e=curve_e.tolist(),
                   fidelity2=np.concatenate(fid_all).mean(0).tolist(),
                   k_star=k_star(curve_n), sec_per_step=t_model / max(n_steps, 1),
                   **rel.result(curve_n, curve_e))
        adj_u, J_u = upper_pairs(adj), upper_pairs(J)
        if gate:
            res["auroc_gate"] = auroc(upper_pairs(np.concatenate(gate)), adj_u)
            res["S_F"] = float(np.concatenate(S_F).mean())
        if attn:
            res["auroc_attn"] = auroc(upper_pairs(np.concatenate(attn)), adj_u)
        if Gn:  # read-out from the effective generator G (true h, J, adj only used here, after the fact)
            Gn = np.concatenate(Gn)
            gu = upper_pairs(Gn)
            res["gen_auroc_adj"] = auroc(gu, adj_u)
            res["gen_corr_J"] = pearson(gu[adj_u], J_u[adj_u])
            res["gen_corr_h"] = pearson(np.diagonal(Gn, axis1=-2, axis2=-1), h)
        if pJ:
            pJ_u = upper_pairs(np.concatenate(pJ))
            res["probe_r2_J"] = r2(pJ_u[adj_u], J_u[adj_u])
            res["probe_r2_h"] = r2(np.concatenate(ph), h)
            res["probe_auroc_adj"] = auroc(upper_pairs(np.concatenate(padj)), adj_u)
        results[T] = res
    model.train()
    return results
