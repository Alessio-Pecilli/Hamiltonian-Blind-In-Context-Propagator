"""Full model: body + Head A (or Head B) + probe. Nothing depends on n or on a site index."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from model.body import Body
from model.embed import at_least_fp32
from model.head_free import FreeHead
from model.head_generator import GeneratorHead
from model.probe import Probe


class Propagator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.head_type = m.get("head", "A")
        self.horizon = cfg["train"]["H"]
        self.body = Body(m["d"], m["de"], m["nh"], m["L"], m.get("conv_kernel", 4),
                         m.get("rope", True), m.get("checkpointing", False),
                         m.get("pair_attn", False), m.get("triangle", False))
        if self.head_type == "A":
            self.head = GeneratorHead(m["d"], m["de"], m["m"], m["K"], m.get("basis_std", 0.02),
                                      m.get("use_S", True),  # absent in old configs = trained with S
                                      m.get("pair_latent", 0), m.get("pauli_A", False),
                                      m.get("pauli_weight", 2), m.get("readout", "linear"))
        elif self.head_type == "B":
            self.head = FreeHead(m["d"], m["de"], self.horizon)
        else:
            raise ValueError(self.head_type)
        # absent in old configs = trained with the probe (keeps old checkpoints loadable)
        self.probe = Probe(m["d"], m["de"]) if m.get("probe", True) else None
        self.amp = m.get("bf16", True)
        self.residual = m.get("residual", False)
        if self.residual and self.head_type != "A":
            raise ValueError("residual anchor needs head A")

    def _autocast(self, device):
        if self.amp and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward(self, node, edge, dt, anchors, T, horizon: int | None = None,
                return_attn: bool = False, keep_feats: bool = False, group: int = 1, perm=None):
        """
        node (B, S, n, 3), edge (B, S, n, n, 9): segments (only positions <= max(T) are read)
        dt (B,), anchors (B, A) context positions for the head, T (B,) last context position.
        group > 1: the windows come in consecutive groups of `group` trajectories of the same Hamiltonian
        (site order perm (B, n)); every window is predicted with the mean generator of its group.
        Returns predictions of shape (B, A, H, n, 3) / (B, A, H, n, n, 9) and diagnostics.
        """
        horizon = horizon or self.horizon
        Tmax = int(T.max())
        node, edge = node[:, : Tmax + 1], edge[:, : Tmax + 1]
        with self._autocast(node.device):
            out = self.body(node, edge, dt, return_attn=return_attn)
        h, e = at_least_fp32(out[0], out[1])
        B, A = anchors.shape
        n = h.shape[2]
        bidx = torch.arange(B, device=h.device)
        hA = h[bidx[:, None], anchors].reshape(B * A, n, -1)
        eA = e[bidx[:, None], anchors].reshape(B * A, n, n, -1)
        dtA = dt[:, None].expand(B, A).reshape(-1)
        o = None
        if self.residual:  # observed frame at each anchor
            o = (node[bidx[:, None], anchors].reshape(B * A, n, 3),
                 edge[bidx[:, None], anchors].reshape(B * A, n, n, 9))
        G = None
        if group > 1:
            if self.head_type != "A":
                raise ValueError("group > 1 needs head A")
            G = self.head.generator(hA, eA)[0].view(B, A, *2 * [hA.shape[1] * self.head.m])
            G = torch.stack([pool_generator(G[:, a], perm, group) for a in range(A)], 1).flatten(0, 1)
        res = self.head(hA, eA, dtA, horizon, o, G=G) if self.head_type == "A" else self.head(hA, eA, dtA, horizon)
        pred = dict(
            node0=node[bidx[:, None], anchors], edge0=edge[bidx[:, None], anchors],  # observed at the anchor
            node=res["node"].view(B, A, horizon, n, 3),
            edge=res["edge"].view(B, A, horizon, n, n, 9),
        )
        if self.head_type == "A":
            pred.update(theta=res["theta"].view(B, A, -1), gate=res["gate"].view(B, A, n, n),
                        S_F=res["S_F"].view(B, A), G=res["G"])
            if "diag_norm" in res:
                pred["diag_norm"] = res["diag_norm"].view(B, A, n)
        if self.probe is not None:
            pred["probe"] = self.probe(h[bidx, T], e[bidx, T])
        if return_attn:
            pred["attn"] = out[2]
        if keep_feats:  # head inputs at the anchors, for cross_predict
            pred.update(hA=hA.view(B, A, n, -1), eA=eA.view(B, A, n, n, -1), dt=dt)
        return pred

    def cross_predict(self, pred: dict, partner: torch.Tensor, perm: torch.Tensor, horizon: int):
        """Predict every window at its first anchor with the generator of another window (partner[b]), whose
        trajectory comes from the same Hamiltonian. The partner's G is moved to this window's site order
        (perm[b, s] = original site of site s). Needs pred from forward(..., keep_feats=True).
        Returns node (B, H, n, 3), edge (B, H, n, n, 9) and the relative generator mismatch (B,)."""
        B, A, n = pred["hA"].shape[:3]
        G = pred["G"].view(B, A, *pred["G"].shape[1:])[:, 0]                     # (B, nm, nm)
        m = G.shape[-1] // n
        inv = torch.argsort(perm, -1)                                             # inv[b, orig] = site
        idx = torch.gather(inv[partner], 1, perm)                                 # partner site of site s
        Gp = G[partner].view(B, n, m, n, m)
        bi = torch.arange(B, device=G.device)[:, None, None]
        Gp = Gp.permute(0, 1, 3, 2, 4)[bi, idx[:, :, None], idx[:, None, :]]      # (B, n, n, m, m)
        Gp = Gp.permute(0, 1, 3, 2, 4).reshape(B, n * m, n * m)
        o = (pred["node0"][:, 0], pred["edge0"][:, 0]) if self.residual else None
        res = self.head(pred["hA"][:, 0], pred["eA"][:, 0], pred["dt"], horizon, o, G=Gp)
        mis = (G - Gp).flatten(1).norm(dim=-1) / G.flatten(1).norm(dim=-1).clamp_min(1e-8)
        return res["node"], res["edge"], mis


def pool_generator(G: torch.Tensor, perm: torch.Tensor, k: int) -> torch.Tensor:
    """G (B, nm, nm) of windows in consecutive groups of k trajectories of the same Hamiltonian, perm (B, n)
    (perm[b, s] = original site of site s) -> the group's mean generator, in each window's own site order."""
    B, nm, _ = G.shape
    n = perm.shape[1]
    m = nm // n
    bi = torch.arange(B, device=G.device)[:, None, None]
    inv = torch.argsort(perm, -1)                                             # inv[b, orig] = site
    Go = G.view(B, n, m, n, m).permute(0, 1, 3, 2, 4)[bi, inv[:, :, None], inv[:, None, :]]  # original order
    Go = Go.reshape(B // k, k, n, n, m, m).mean(1).repeat_interleave(k, 0)
    return Go[bi, perm[:, :, None], perm[:, None, :]].permute(0, 1, 3, 2, 4).reshape(B, nm, nm)


def block_rollout(model, node, edge, dt, anchors, T, horizon: int, block: int, p_pred: float = 1.0,
                  return_attn: bool = False, keep_feats: bool = False, **kw):
    """Predict `horizon` steps in blocks of `block`: after each block its frames are written into the
    context and the next block is predicted from T + block. Per example, the written frames are the
    model's own (detached) predictions with probability p_pred, else the true ones (scheduled sampling;
    p_pred = 1 is the free roll out used at test time). node/edge must hold the true frames up to T + horizon.
    kw (group, perm) go to every model call. Yields (pred, anchors) per block; anchors[:, 0] must be T (true for the dataset windows)."""
    node, edge, T = node.clone(), edge.clone(), T.clone()
    B = T.shape[0]
    bidx = torch.arange(B, device=T.device)[:, None]
    for r0 in range(0, horizon, block):
        hb = min(block, horizon - r0)
        pred = model(node, edge, dt, anchors, T, hb, return_attn=return_attn and r0 == 0,
                     keep_feats=keep_feats and r0 == 0, **kw)
        yield pred, anchors
        idx = T[:, None] + torch.arange(1, hb + 1, device=T.device)       # (B, hb)
        own = torch.rand(B, device=T.device) < p_pred
        for x, p in ((node, pred["node"]), (edge, pred["edge"])):
            m = own.view(B, *([1] * (p.dim() - 2)))
            x[bidx, idx] = torch.where(m, p[:, 0].detach().to(x.dtype), x[bidx, idx])
        T = T + hb
        anchors = T[:, None]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
