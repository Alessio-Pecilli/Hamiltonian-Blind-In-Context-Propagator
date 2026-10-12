"""Body (Sec. 3.5): L blocks of (spatial layer -> temporal layer), final LayerNorm on h and e."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.embed import Embedding, offdiag_mask
from model.spatial import SpatialLayer
from model.temporal import TemporalLayer


class Block(nn.Module):
    def __init__(self, d, de, nh, kernel=4, use_rope=True, pair_attn=False, triangle=False):
        super().__init__()
        self.spatial = SpatialLayer(d, de, nh, triangle)
        self.temporal = TemporalLayer(d, de, nh, kernel, use_rope, pair_attn)

    def forward(self, h, e, return_attn=False):
        if return_attn:
            h, e, a = self.spatial(h, e, return_attn=True)
            h, e = self.temporal(h, e)
            return h, e, a
        h, e = self.spatial(h, e)
        return self.temporal(h, e)


class Body(nn.Module):
    def __init__(self, d, de, nh, L, kernel=4, use_rope=True, checkpointing=False,
                 pair_attn=False, triangle=False):
        super().__init__()
        self.embed = Embedding(d, de)
        self.blocks = nn.ModuleList([Block(d, de, nh, kernel, use_rope, pair_attn, triangle) for _ in range(L)])
        self.ln_h = nn.LayerNorm(d)
        self.ln_e = nn.LayerNorm(de)
        self.checkpointing = checkpointing

    def forward(self, node, edge, dt, return_attn: bool = False):
        """node (B,T,n,3), edge (B,T,n,n,9), dt (B,) ->
        h (B,T,n,d), e (B,T,n,n,de) [, attn: list of (B,T,nh,n,n) per block]."""
        h, e = self.embed(node, edge, dt)
        attns = []
        for blk in self.blocks:
            if return_attn:
                h, e, a = blk(h, e, return_attn=True)
                attns.append(a)
            elif self.checkpointing and self.training:
                h, e = checkpoint(blk, h, e, use_reentrant=False)
            else:
                h, e = blk(h, e)
        n = h.shape[-2]
        h = self.ln_h(h)
        e = self.ln_e(e) * offdiag_mask(n, e.device)[..., None]
        return (h, e, attns) if return_attn else (h, e)
