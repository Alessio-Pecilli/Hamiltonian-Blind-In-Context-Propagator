"""Collect everything the HBICP dashboard shows into one JSON file.

    python -m tools.export_dashboard \
        --runs runs --results results \
        --data datasets/n8/test.h5 --ckpt runs/local_A/last.pt \
        --T 32 --horizon 64 --sample 0 --out dashboard_data.json

Then open the dashboard and press "Carica dashboard_data.json", or publish it on GitHub Pages
with tools.publish_dashboard.

What is exported
  * runs/<name>/config.json + log.jsonl       -> config, parameter count, train/val curves
  * results/<name>/results.json               -> RMSE_n,e(k), k*(T), AUROC, probe, S_F, cost
  * every HDF5 file passed with --data        -> dataset attributes and shapes
  * one test trajectory (--sample)            -> context, true future, predictions of the model
                                                 and of the baselines, H / graph metadata,
                                                 gate g_ij, probe, spectrum of G_t,
                                                 energy and lambda_min(rho_ij) along the rollout
Everything is optional: missing pieces are simply left out of the JSON.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os

import h5py
import numpy as np

from config import git_commit


def _r(x, nd=5):
    """Round nested arrays to keep the JSON small."""
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, bytes):
        return o.decode()
    return o


# ----------------------------------------------------------------------------------------
# (regex on the state_dict key, group id). First match wins. Groups follow the node / pair paths of
# the body so the dashboard can show where the capacity sits.
_GROUPS = [
    (r"^body\.embed\.mlp_n", "embed_node"),
    (r"^body\.embed\.mlp_e", "embed_pair"),
    (r"^body\.blocks\.\d+\.spatial\.(w_b|w_ev)", "spatial_pair_to_node"),
    (r"^body\.blocks\.\d+\.spatial\.(ln_h2|ln_e2|w_a|w_bb|mlp_eu)", "spatial_pair_update"),
    (r"^body\.blocks\.\d+\.spatial\.", "spatial_node"),
    (r"^body\.blocks\.\d+\.temporal\.(conv|ln_e)", "temporal_pair"),
    (r"^body\.blocks\.\d+\.temporal\.", "temporal_node"),
    (r"^body\.ln_h", "final_ln_node"),
    (r"^body\.ln_e", "final_ln_pair"),
    (r"^head\.(w_g|w_A)", "head_pair"),
    (r"^head\.", "head_node_and_generator"),
    (r"^probe\.", "probe"),
]


def param_groups(sd) -> dict:
    """Number of parameters of a state_dict per functional group (see _GROUPS)."""
    import re
    out: dict[str, int] = {}
    for k, v in sd.items():
        g = next((g for pat, g in _GROUPS if re.match(pat, k)), "other")
        out[g] = out.get(g, 0) + int(v.numel())
    return out


def load_runs(root: str, max_points: int = 400) -> list[dict]:
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        train, val = [], []
        log_path = os.path.join(d, "log.jsonl")
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    (val if rec.get("split") == "val" else train).append(rec)
        if len(train) > max_points:  # keep first, last and an even subsample
            idx = np.unique(np.linspace(0, len(train) - 1, max_points).astype(int))
            train = [train[i] for i in idx]
        params, groups = None, None
        ck = os.path.join(d, "last.pt")
        if os.path.exists(ck):
            try:
                import torch
                sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
                params = int(sum(v.numel() for v in sd.values()))
                groups = param_groups(sd)
            except Exception:
                pass
        runs.append(dict(name=os.path.basename(d), path=d, config=cfg, params=params, param_groups=groups,
                         train=train, val=val,
                         mtime=_dt.datetime.fromtimestamp(os.path.getmtime(cfg_path)).isoformat()))
    return runs


def load_results(root: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        with open(p) as f:
            out.append(dict(name=os.path.basename(os.path.dirname(p)), path=p, methods=json.load(f)))
    return out


def dataset_info(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return dict(path=path, attrs={k: _clean(v) for k, v in f.attrs.items()},
                    shapes={k: list(f[k].shape) for k in f.keys()},
                    mean_degree=float(np.asarray(f["adj"][:]).sum(-1).mean()),
                    J_on_edges=_r([np.asarray(f["J"][:])[np.asarray(f["adj"][:])].mean(),
                                   np.asarray(f["J"][:])[np.asarray(f["adj"][:])].std()], 4),
                    h_stats=_r([np.asarray(f["h"][:]).mean(), np.asarray(f["h"][:]).std()], 4))


# ----------------------------------------------------------------------------------------
def tfim_energy(node, edge, J, h):
    """<H> = -sum_{i<j} J_ij <Z_i Z_j> - sum_i h_i <X_i> from observables (any leading dims)."""
    iu, ju = np.triu_indices(len(h), 1)
    zz = edge[..., iu, ju, 8]
    return -(zz * J[iu, ju]).sum(-1) - (node[..., :, 0] * h).sum(-1)


def lam_min(node, edge):
    """min over pairs of lambda_min(rho_ij) for each time step: (K, n, 3), (K, n, n, 9) -> (K,)."""
    from data.physics import PAULIS, SI
    ps = (SI,) + PAULIS
    P16 = np.stack([np.stack([np.kron(a, b) for b in ps]) for a in ps])
    n = node.shape[-2]
    iu, ju = np.triu_indices(n, 1)
    K = node.shape[0]
    c = np.zeros((K, len(iu), 4, 4))
    c[..., 0, 0] = 1
    c[..., 1:, 0] = node[:, iu]
    c[..., 0, 1:] = node[:, ju]
    c[..., 1:, 1:] = edge[:, iu, ju].reshape(K, len(iu), 3, 3)
    rho = 0.25 * np.einsum("kpab,abxy->kpxy", c, P16)
    return np.linalg.eigvalsh(rho)[..., 0].min(-1)


def sample_block(data, idx, T, horizon, ckpt, baselines, device="cpu"):
    from eval.baselines import BASELINES
    from eval.metrics import fidelity_pairs, rmse_per_traj
    with h5py.File(data, "r") as f:
        attrs = dict(f.attrs)
        node, edge = f["node"][idx], f["edge"][idx]
        J, h, adj, seed, dt = f["J"][idx], f["h"][idx], f["adj"][idx], f["seed"][idx], f["dt"][idx]
    L, n = node.shape[:2]
    Hk = min(horizon, L - 1 - T)
    ctx_n, ctx_e = node[: T + 1], edge[: T + 1]
    tgt_n, tgt_e = node[T + 1: T + 1 + Hk], edge[T + 1: T + 1 + Hk]
    preds, extra = {}, {}
    if ckpt:
        import torch
        from data.dataset import collate, gather_targets  # noqa: F401
        from eval.evaluate import load_model
        model, ck = load_model(ckpt, torch.device(device))
        H_model = Hk if model.head_type == "A" else min(Hk, model.horizon)
        b = dict(node=torch.tensor(node[None, : T + 1 + H_model]),
                 edge=torch.tensor(edge[None, : T + 1 + H_model]),
                 dt=torch.tensor([dt], dtype=torch.float32), anchors=torch.tensor([[T]]),
                 T=torch.tensor([T]))
        with torch.no_grad():
            p = model(b["node"], b["edge"], b["dt"], b["anchors"], b["T"], horizon=H_model,
                      return_attn=True)
        name = f"model({model.head_type})"
        preds[name] = (p["node"][0, 0].numpy(), p["edge"][0, 0].numpy())
        extra["probe"] = dict(h=_r(p["probe"]["h"][0].numpy(), 4), J=_r(p["probe"]["J"][0].numpy(), 4),
                              adj=_r(torch.sigmoid(p["probe"]["adj_logit"][0]).numpy(), 4))
        if "gate" in p:
            extra["gate"] = _r(p["gate"][0, 0].numpy(), 4)
            ev = np.linalg.eigvals(p["G"][0].double().numpy())
            extra["G_eig"] = dict(re=_r(ev.real, 5), im=_r(ev.imag, 5),
                                  max_re=float(ev.real.max()), S_F=float(p["S_F"][0, 0]))
            extra["theta"] = _r(p["theta"][0, 0].numpy(), 4)
        if "attn" in p:
            a = torch.stack([x.float().mean((1, 2)) for x in p["attn"]]).mean(0)[0]
            extra["attn"] = _r((0.5 * (a + a.T)).numpy(), 4)
    for bname in baselines:
        try:
            preds[bname] = BASELINES[bname](ctx_n, ctx_e, Hk, attrs=attrs, seed=seed, t_abs=T)
        except Exception as ex:  # e.g. hl_oracle on a non-TFIM family
            print(f"  skip {bname}: {ex}")
    out_preds = {}
    for k, (pn, pe) in preds.items():
        K = pn.shape[0]
        rn, re = rmse_per_traj(pn[None], tgt_n[None, :K], pe[None], tgt_e[None, :K])
        fid = fidelity_pairs(pn[None], tgt_n[None, :K], pe[None], tgt_e[None, :K])[0]
        d = dict(node=_r(pn, 4), rmse_n=_r(rn[0], 5), rmse_e=_r(re[0], 5), fid2=_r(fid, 5),
                 lam_min=_r(lam_min(pn.astype(np.float64), pe.astype(np.float64)), 5))
        # edge predictions only for ZZ, XX and YZ channels to keep the file small
        d["edge_sel"] = {c: _r(pe[..., ci], 4) for c, ci in (("xx", 0), ("yz", 5), ("zz", 8))}
        if attrs.get("family") in ("tfim", "lr_ising"):
            d["energy"] = _r(tfim_energy(pn, pe, J, h), 5)
        out_preds[k] = d
    full_n, full_e = node[: T + 1 + Hk], edge[: T + 1 + Hk]
    s = dict(dataset=data, index=int(idx), seed=int(seed), n=int(n), dt=float(dt), T=int(T), H=int(Hk),
             family=str(attrs.get("family")), J=_r(J, 4), h=_r(h, 4), adj=np.asarray(adj).astype(int).tolist(),
             node=_r(full_n, 4),
             edge_sel={c: _r(full_e[..., ci], 4) for c, ci in (("xx", 0), ("yz", 5), ("zz", 8))},
             lam_min_true=_r(lam_min(full_n.astype(np.float64), full_e.astype(np.float64)), 5),
             preds=out_preds, **extra)
    if attrs.get("family") in ("tfim", "lr_ising"):
        s["energy_true"] = _r(tfim_energy(full_n, full_e, J, h), 5)
    return s


def add_export_args(ap: argparse.ArgumentParser) -> None:
    """Arguments selecting what goes into the export (shared with tools.publish_dashboard)."""
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", nargs="*", default=[], help="HDF5 files; the first is used for the sample")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--baselines", nargs="*", default=["persistence", "dmd", "dmd4", "oracle", "hl_oracle"])


def build_export(a: argparse.Namespace, notes: str = "") -> dict:
    """Runs, results, datasets and the sample rollout selected by add_export_args, as the
    JSON-ready dict of the dashboard (schema hbicp-dashboard/1)."""
    out = dict(schema="hbicp-dashboard/1", generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
               git_commit=git_commit(), notes=notes,
               runs=load_runs(a.runs) if os.path.isdir(a.runs) else [],
               results=load_results(a.results) if os.path.isdir(a.results) else [],
               datasets=[dataset_info(p) for p in a.data if os.path.exists(p)])
    if a.data and os.path.exists(a.data[0]):
        out["sample"] = sample_block(a.data[0], a.sample, a.T, a.horizon, a.ckpt, a.baselines)
    return _clean(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_export_args(ap)
    ap.add_argument("--notes", default="", help="free text shown on the dashboard")
    ap.add_argument("--out", default="dashboard_data.json")
    a = ap.parse_args()
    out = build_export(a, a.notes)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB): {len(out['runs'])} runs, "
          f"{len(out['results'])} result sets, {len(out['datasets'])} datasets")


if __name__ == "__main__":
    main()
