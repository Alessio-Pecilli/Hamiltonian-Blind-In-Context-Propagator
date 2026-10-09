"""Head A: in-context structured generator (Sec. 3.6).

G_t = (A - A^T) - S S^T  with  A_ii = sum_k theta_k A_k,  A_ij = g_ij reshape(W_A e_ij),
S = blockdiag(sum_k theta_k S_k).  Re eig(G) <= 0, hence ||exp(k dt G)||_2 <= 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.embed import at_least_fp32


def symmetrize_edge_channels(x: torch.Tensor) -> torch.Tensor:
    """x (..., n, n, 9) -> average of x[i,j,(a,b)] and x[j,i,(b,a)] (dims -3,-2 are sites)."""
    s = x.shape
    x3 = x.reshape(*s[:-1], 3, 3)
    xt = x3.transpose(-3, -4).transpose(-1, -2)
    return (0.5 * (x3 + xt)).reshape(s)


class GeneratorHead(nn.Module):
    def __init__(self, d: int, de: int, m: int, K: int, basis_std: float = 0.02):
        super().__init__()
        self.m, self.K = m, K
        self.w_theta = nn.Linear(d, K)
        self.w_z = nn.Linear(d, m)
        self.A_basis = nn.Parameter(torch.randn(K, m, m) * basis_std)
        self.S_basis = nn.Parameter(torch.randn(K, m, m) * basis_std)
        self.w_A = nn.Linear(de, m * m)
        nn.init.zeros_(self.w_A.weight)  # start with uncoupled sites
        nn.init.zeros_(self.w_A.bias)
        self.w_g = nn.Linear(de, 1)
        # decoders
        self.D_n = nn.Linear(m, 3)
        self.B = nn.Parameter(torch.randn(9, m, m) / m)
        self.beta = nn.Parameter(torch.zeros(9))

    # ------------------------------------------------------------------
    def generator(self, h: torch.Tensor, e: torch.Tensor):
        """h (N, n, d), e (N, n, n, de) -> G (N, nm, nm), theta (N, K), gate (N, n, n), S_F (N,)."""
        N, n, _ = h.shape
        m = self.m
        theta = self.w_theta(h.mean(1))                                       # (N, K)
        A_diag = torch.einsum("nk,kab->nab", theta, self.A_basis)             # (N, m, m)
        gate = torch.sigmoid(self.w_g(e).squeeze(-1))                         # (N, n, n)
        eye = torch.eye(n, device=h.device, dtype=torch.bool)
        gate = gate.masked_fill(eye, 0.0)
        A_off = gate[..., None, None] * self.w_A(e).view(N, n, n, m, m)       # (N, n, n, m, m)
        A_blocks = torch.where(eye[None, :, :, None, None], A_diag[:, None, None], A_off)
        A = A_blocks.permute(0, 1, 3, 2, 4).reshape(N, n * m, n * m)
        S_b = torch.einsum("nk,kab->nab", theta, self.S_basis)                # (N, m, m)
        SS_b = S_b @ S_b.transpose(-1, -2)
        SS = torch.einsum("ij,nab->niajb", eye.to(h.dtype), SS_b).reshape(N, n * m, n * m)
        G = (A - A.transpose(-1, -2)) - SS
        S_F = S_b.flatten(1).norm(dim=-1) * n ** 0.5                          # ||blockdiag||_F
        return G, theta, gate, S_F

    def decode(self, z: torch.Tensor):
        """z (..., n, m) -> node (..., n, 3), edge (..., n, n, 9)."""
        node = torch.tanh(self.D_n(z))
        zB = torch.einsum("...im,cmk->...ick", z, self.B)
        x = torch.einsum("...ick,...jk->...ijc", zB, z)
        edge = torch.tanh(symmetrize_edge_channels(x) + self.beta)
        return node, edge

    def forward(self, h: torch.Tensor, e: torch.Tensor, dt: torch.Tensor, horizon: int):
        """h (N, n, d), e (N, n, n, de), dt (N,) ->
        node (N, H, n, 3), edge (N, H, n, n, 9) + diagnostics. Runs in float32."""
        h, e, dt = at_least_fp32(h, e, dt)
        N, n, _ = h.shape
        G, theta, gate, S_F = self.generator(h, e)
        P = torch.linalg.matrix_exp(dt[:, None, None] * G)
        Z = self.w_z(h).reshape(N, n * self.m)
        zs = []
        for _ in range(horizon):  # repeated mat-vec, never powers of P
            Z = (P @ Z[..., None]).squeeze(-1)
            zs.append(Z)
        z = torch.stack(zs, 1).view(N, horizon, n, self.m)
        node, edge = self.decode(z)
        return dict(node=node, edge=edge, theta=theta, gate=gate, S_F=S_F, G=G, z=z)
