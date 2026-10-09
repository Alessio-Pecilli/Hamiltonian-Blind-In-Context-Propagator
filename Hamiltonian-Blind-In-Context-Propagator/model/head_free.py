"""Head B: free multi-horizon readout (Sec. 3.7, ablation only)."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.embed import at_least_fp32, mlp
from model.head_generator import symmetrize_edge_channels


class FreeHead(nn.Module):
    def __init__(self, d: int, de: int, horizon: int):
        super().__init__()
        self.H = horizon
        self.mlp_n = mlp(d, 4 * d, horizon * 3)
        self.w_a = nn.Linear(d, de)
        self.w_b = nn.Linear(d, de)
        self.mlp_e = mlp(2 * de, 4 * de, horizon * 9)

    def forward(self, h: torch.Tensor, e: torch.Tensor, dt=None, horizon: int | None = None):
        """h (N, n, d), e (N, n, n, de) -> node (N, H, n, 3), edge (N, H, n, n, 9)."""
        if horizon is not None and horizon != self.H:
            raise ValueError("Head B has a fixed horizon")
        h, e = at_least_fp32(h, e)
        N, n, _ = h.shape
        node = torch.tanh(self.mlp_n(h).view(N, n, self.H, 3)).transpose(1, 2)
        prod = self.w_a(h)[:, :, None] * self.w_b(h)[:, None, :]
        x = self.mlp_e(torch.cat([e, prod], -1)).view(N, n, n, self.H, 9)
        edge = torch.tanh(symmetrize_edge_channels(x.permute(0, 3, 1, 2, 4)))
        return dict(node=node, edge=edge)
