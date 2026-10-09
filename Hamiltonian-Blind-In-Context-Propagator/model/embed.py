"""Embeddings (Sec. 3.2). No absolute site index is ever embedded."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def at_least_fp32(*xs):
    """Cast half/bfloat16 tensors to float32 (the head is numerically sensitive); keep float64."""
    out = tuple(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x for x in xs)
    return out if len(out) > 1 else out[0]


def mlp(din: int, dhidden: int, dout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dhidden), nn.GELU(), nn.Linear(dhidden, dout))


def sinusoidal_pe(T: int, d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Standard sinusoidal encoding of the window-relative index t, omega_k = 10000^(-2k/d)."""
    t = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    k = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    w = torch.exp(-math.log(10000.0) * k / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(t * w)
    pe[:, 1::2] = torch.cos(t * w[: pe[:, 1::2].shape[1]])
    return pe.to(dtype)


def offdiag_mask(n: int, device=None) -> torch.Tensor:
    return ~torch.eye(n, dtype=torch.bool, device=device)


class Embedding(nn.Module):
    def __init__(self, d: int, de: int):
        super().__init__()
        self.d = d
        self.mlp_n = mlp(4, 4 * d, d)    # [N_t[i,:], log dt] -> R^d
        self.mlp_e = mlp(9, 4 * de, de)  # E_t[i,j,:] -> R^de

    def forward(self, node: torch.Tensor, edge: torch.Tensor, dt: torch.Tensor):
        """node (B,T,n,3), edge (B,T,n,n,9), dt (B,) -> h (B,T,n,d), e (B,T,n,n,de)."""
        B, T, n, _ = node.shape
        logdt = torch.log(dt).to(node.dtype)[:, None, None, None].expand(B, T, n, 1)
        h = self.mlp_n(torch.cat([node, logdt], -1))
        h = h + sinusoidal_pe(T, self.d, node.device, h.dtype)[None, :, None, :]
        e = self.mlp_e(edge) * offdiag_mask(n, node.device)[None, None, :, :, None]
        return h, e
