"""Metrics (Sec. 6): horizon RMSE, predictability horizon k*, graph recovery, probe quality,
dissipation gauge, cost."""
from __future__ import annotations

import time

import h5py
import numpy as np
import torch

from data.dataset import gather_targets, make_loader

EPS_KSTAR = 0.05


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
        rn_all, re_all, gate, attn, adj, J, h = [], [], [], [], [], [], []
        pJ, ph, padj, S_F = [], [], [], []
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
            adj.append(batch["adj"].cpu().numpy())
            J.append(batch["J"].cpu().numpy())
            h.append(batch["h"].cpu().numpy())
            pJ.append(pred["probe"]["J"].cpu().numpy())
            ph.append(pred["probe"]["h"].cpu().numpy())
            padj.append(pred["probe"]["adj_logit"].cpu().numpy())
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
                   k_star=k_star(curve_n), sec_per_step=t_model / max(n_steps, 1))
        adj_u, J_u = upper_pairs(adj), upper_pairs(J)
        if gate:
            res["auroc_gate"] = auroc(upper_pairs(np.concatenate(gate)), adj_u)
            res["S_F"] = float(np.concatenate(S_F).mean())
        if attn:
            res["auroc_attn"] = auroc(upper_pairs(np.concatenate(attn)), adj_u)
        pJ_u = upper_pairs(np.concatenate(pJ))
        res["probe_r2_J"] = r2(pJ_u[adj_u], J_u[adj_u])
        res["probe_r2_h"] = r2(np.concatenate(ph), h)
        res["probe_auroc_adj"] = auroc(upper_pairs(np.concatenate(padj)), adj_u)
        results[T] = res
    model.train()
    return results
