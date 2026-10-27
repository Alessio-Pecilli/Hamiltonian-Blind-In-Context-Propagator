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
                return_attn: bool = False):
        """
        node (B, S, n, 3), edge (B, S, n, n, 9): segments (only positions <= max(T) are read)
        dt (B,), anchors (B, A) context positions for the head, T (B,) last context position.
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
        res = self.head(hA, eA, dtA, horizon, o) if self.residual else self.head(hA, eA, dtA, horizon)
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
        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
