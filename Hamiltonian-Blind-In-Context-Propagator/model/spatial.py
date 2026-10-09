"""Spatial layer (Sec. 3.3): attention among the n sites at fixed t, pair-biased and
value-augmented by pair features, followed by the pair-feature update (Eq. 13)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.embed import mlp, offdiag_mask


class SpatialLayer(nn.Module):
    def __init__(self, d: int, de: int, nh: int):
        super().__init__()
        assert d % nh == 0
        self.d, self.de, self.nh, self.dh = d, de, nh, d // nh
        self.ln_h = nn.LayerNorm(d)
        self.ln_e = nn.LayerNorm(de)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.w_b = nn.Linear(de, nh, bias=False)                      # pair bias, Eq. (8)
        self.w_ev = nn.Parameter(torch.empty(nh, de, self.dh))        # value augmentation, Eq. (10)
        nn.init.uniform_(self.w_ev, -de ** -0.5, de ** -0.5)  # as nn.Linear(de, dh)
        self.w_o = nn.Linear(d, d)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = mlp(d, 4 * d, d)
        # pair update, Eq. (13)
        self.ln_h2 = nn.LayerNorm(d)
        self.ln_e2 = nn.LayerNorm(de)
        self.w_a = nn.Linear(d, de)
        self.w_bb = nn.Linear(d, de)
        self.mlp_eu = mlp(2 * de, 4 * de, de)

    def forward(self, h: torch.Tensor, e: torch.Tensor, return_attn: bool = False):
        """h (..., n, d), e (..., n, n, de). Leading dims are independent (batch, time)."""
        *lead, n, d = h.shape
        mask = offdiag_mask(n, h.device)[..., None].to(h.dtype)  # keep e_ii = 0
        x = self.ln_h(h)
        le = self.ln_e(e) * mask
        q, k, v = self.qkv(x).view(*lead, n, 3, self.nh, self.dh).unbind(-3)
        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))           # (..., nh, n, dh)
        bias = self.w_b(le).movedim(-1, -3)                          # (..., nh, n, n)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + bias
        alpha = logits.softmax(-1)                                   # (..., nh, n, n)
        # sum_j alpha_ij (v_j + W_ev LN(e_ij)) = alpha @ v + W_ev (sum_j alpha_ij LN(e_ij))
        ebar = torch.einsum("...hij,...ijc->...hic", alpha, le)      # (..., nh, n, de)
        u = alpha @ v + torch.einsum("...hic,hcd->...hid", ebar, self.w_ev.to(ebar.dtype))
        u = u.transpose(-2, -3).reshape(*lead, n, d)
        h = h + self.w_o(u)
        h = h + self.ff(self.ln_ff(h))
        # pair-feature update from the sites it connects
        x = self.ln_h2(h)
        prod = self.w_a(x)[..., :, None, :] * self.w_bb(x)[..., None, :, :]  # (..., n, n, de)
        e = e + self.mlp_eu(torch.cat([self.ln_e2(e), prod], -1))
        e = 0.5 * (e + e.transpose(-2, -3)) * mask                   # symmetrise, e_ii = 0
        return (h, e, alpha) if return_attn else (h, e)
