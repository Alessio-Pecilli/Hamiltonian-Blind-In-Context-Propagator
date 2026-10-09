"""Probe head (Sec. 3.8): linear read-out of (h_i, J_ij, adj_ij) on stop-gradient features."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.embed import at_least_fp32


class Probe(nn.Module):
    def __init__(self, d: int, de: int):
        super().__init__()
        self.w_h = nn.Linear(d, 1)
        self.w_J = nn.Linear(de, 1)
        self.w_adj = nn.Linear(de, 1)

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """h (B, n, d), e (B, n, n, de) at the last context position; gradients never reach the body."""
        h, e = at_least_fp32(h.detach(), e.detach())
        return dict(
            h=self.w_h(h).squeeze(-1),
            J=self.w_J(e).squeeze(-1),
            adj_logit=self.w_adj(e).squeeze(-1),
        )
