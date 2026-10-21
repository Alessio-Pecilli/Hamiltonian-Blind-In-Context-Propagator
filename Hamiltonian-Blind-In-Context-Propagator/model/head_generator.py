"""Head A: in-context structured generator (Sec. 3.6).

G_t = (A - A^T) - S S^T  with  A_ii = sum_k theta_k A_k,  A_ij = g_ij reshape(W_A e_ij),
S = blockdiag(sum_k theta_k S_k).  Re eig(G) <= 0, hence ||exp(k dt G)||_2 <= 1.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn

from model.embed import at_least_fp32


def pauli_basis(m: int, weight: int = 2) -> torch.Tensor:
    """Fixed real basis (S, m, m) of Pauli strings on q = log2(m) qubits with weight <= `weight`
    (0 = all 4^q). Strings with an odd number of Y are imaginary: they are multiplied by i, which
    makes every element real. Elements are Frobenius-orthogonal with ||P||_F^2 = m."""
    q = int(np.log2(m))
    if 2 ** q != m:
        raise ValueError(f"pauli_A needs m to be a power of 2, got m={m}")
    P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
    out = []
    for idx in itertools.product(range(4), repeat=q):
        if weight and sum(i > 0 for i in idx) > weight:
            continue
        M = np.eye(1)
        for i in idx:
            M = np.kron(M, P[i])
        out.append((M * 1j if idx.count(2) % 2 else M).real)
    return torch.tensor(np.stack(out), dtype=torch.float32)


def symmetrize_edge_channels(x: torch.Tensor) -> torch.Tensor:
    """x (..., n, n, 9) -> average of x[i,j,(a,b)] and x[j,i,(b,a)] (dims -3,-2 are sites)."""
    s = x.shape
    x3 = x.reshape(*s[:-1], 3, 3)
    xt = x3.transpose(-3, -4).transpose(-1, -2)
    return (0.5 * (x3 + xt)).reshape(s)


class GeneratorHead(nn.Module):
    def __init__(self, d: int, de: int, m: int, K: int, basis_std: float = 0.02, use_S: bool = True,
                 pair_latent: int = 0, pauli_A: bool = False, pauli_weight: int = 2):
        super().__init__()
        self.m, self.K, self.use_S, self.p, self.pauli_A = m, K, use_S, pair_latent, pauli_A
        self.w_z = nn.Linear(d, m)
        if pauli_A:
            if use_S:
                raise ValueError("pauli_A is only implemented without dissipation (use_S=false)")
            self.register_buffer("pauli", pauli_basis(m, pauli_weight))  # fixed, not learned
            self.coef = nn.Sequential(nn.Linear(2 * d + de, d), nn.GELU(), nn.Linear(d, len(self.pauli)))
            nn.init.zeros_(self.coef[-1].weight)  # start with G = 0
            nn.init.zeros_(self.coef[-1].bias)
        else:
            self.w_theta = nn.Linear(d, K)
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
        if pair_latent:  # per-pair latent u_ij in R^p, rotated by its own skew generator C_ij(e_ij)
            self.w_u = nn.Linear(de, pair_latent)
            self.w_C = nn.Linear(de, pair_latent * pair_latent)
            nn.init.zeros_(self.w_C.weight)  # start with frozen pair latents (C = 0)
            nn.init.zeros_(self.w_C.bias)
            self.D_u = nn.Linear(pair_latent, 9)
            nn.init.zeros_(self.D_u.weight)  # start as the model without pair latent
            nn.init.zeros_(self.D_u.bias)

    # ------------------------------------------------------------------
    def generator(self, h: torch.Tensor, e: torch.Tensor):
        """h (N, n, d), e (N, n, n, de) -> G (N, nm, nm), theta (N, K), gate (N, n, n), S_F (N,)."""
        N, n, _ = h.shape
        m = self.m
        if self.pauli_A:
            return self._pauli_generator(h, e)
        theta = self.w_theta(h.mean(1))                                       # (N, K)
        A_diag = torch.einsum("nk,kab->nab", theta, self.A_basis)             # (N, m, m)
        gate = torch.sigmoid(self.w_g(e).squeeze(-1))                         # (N, n, n)
        eye = torch.eye(n, device=h.device, dtype=torch.bool)
        gate = gate.masked_fill(eye, 0.0)
        A_off = gate[..., None, None] * self.w_A(e).view(N, n, n, m, m)       # (N, n, n, m, m)
        A_blocks = torch.where(eye[None, :, :, None, None], A_diag[:, None, None], A_off)
        A = A_blocks.permute(0, 1, 3, 2, 4).reshape(N, n * m, n * m)
        G = A - A.transpose(-1, -2)
        if not self.use_S:  # closed-system ablation: G is skew-symmetric, ||exp(k dt G)||_2 = 1
            return G, theta, gate, h.new_zeros(N)
        S_b = torch.einsum("nk,kab->nab", theta, self.S_basis)                # (N, m, m)
        SS_b = S_b @ S_b.transpose(-1, -2)
        SS = torch.einsum("ij,nab->niajb", eye.to(h.dtype), SS_b).reshape(N, n * m, n * m)
        G = G - SS
        S_F = S_b.flatten(1).norm(dim=-1) * n ** 0.5                          # ||blockdiag||_F
        return G, theta, gate, S_F

    def _pauli_generator(self, h: torch.Tensor, e: torch.Tensor):
        """Every block A_ij = sum_s c_s P_s with one MLP c = f([h_i, h_j, e_ij]) (e_ii = 0).
        Returns G, theta (zeros), gate = ||A_ij||_F / max_{i!=j}, diag_norm = ||A_ii||_F, S_F (zeros)."""
        N, n, _ = h.shape
        m = self.m
        eye = torch.eye(n, device=h.device, dtype=torch.bool)
        e = e.masked_fill(eye[None, :, :, None], 0.0)
        hi, hj = h[:, :, None].expand(N, n, n, -1), h[:, None].expand(N, n, n, -1)
        c = self.coef(torch.cat([hi, hj, e], -1))                             # (N, n, n, S)
        A_blocks = torch.einsum("nijs,sab->nijab", c, self.pauli)
        A = A_blocks.permute(0, 1, 3, 2, 4).reshape(N, n * m, n * m)
        G = A - A.transpose(-1, -2)
        norm = (m * (c ** 2).sum(-1)).sqrt()                                  # ||A_ij||_F (orthogonal basis)
        off = norm.masked_fill(eye, 0.0)
        gate = off / off.amax((-1, -2), keepdim=True).clamp_min(1e-12)
        return G, h.new_zeros(N, 1), gate, h.new_zeros(N), norm.diagonal(dim1=-2, dim2=-1)

    def decode(self, z: torch.Tensor, u: torch.Tensor | None = None):
        """z (..., n, m) [, pair latents u (..., n, n, p)] -> node (..., n, 3), edge (..., n, n, 9)."""
        node = torch.tanh(self.D_n(z))
        zB = torch.einsum("...im,cmk->...ick", z, self.B)
        x = torch.einsum("...ick,...jk->...ijc", zB, z)
        if u is not None:
            x = x + self.D_u(u)
        edge = torch.tanh(symmetrize_edge_channels(x + self.beta))  # beta inside: symmetry holds exactly
        return node, edge

    def forward(self, h: torch.Tensor, e: torch.Tensor, dt: torch.Tensor, horizon: int):
        """h (N, n, d), e (N, n, n, de), dt (N,) ->
        node (N, H, n, 3), edge (N, H, n, n, 9) + diagnostics. Runs in float32."""
        h, e, dt = at_least_fp32(h, e, dt)
        N, n, _ = h.shape
        G, theta, gate, S_F, *diag = self.generator(h, e)
        P = torch.linalg.matrix_exp(dt[:, None, None] * G)
        Z = self.w_z(h).reshape(N, n * self.m)
        zs = []
        for _ in range(horizon):  # repeated mat-vec, never powers of P
            Z = (P @ Z[..., None]).squeeze(-1)
            zs.append(Z)
        z = torch.stack(zs, 1).view(N, horizon, n, self.m)
        u = self.pair_rollout(e, dt, horizon) if self.p else None
        node, edge = self.decode(z, u)
        out = dict(node=node, edge=edge, theta=theta, gate=gate, S_F=S_F, G=G, z=z)
        if diag:
            out["diag_norm"] = diag[0]
        return out

    def pair_rollout(self, e: torch.Tensor, dt: torch.Tensor, horizon: int) -> torch.Tensor:
        """u_ij(k) = exp(k dt C_ij) u_ij(0), C_ij = B_ij - B_ij^T (stable): (N, H, n, n, p)."""
        N, n, _, _ = e.shape
        p = self.p
        Bm = self.w_C(e).view(N, n, n, p, p)
        C = Bm - Bm.transpose(-1, -2)
        P = torch.linalg.matrix_exp(dt[:, None, None, None, None] * C)
        U = self.w_u(e)
        us = []
        for _ in range(horizon):
            U = (P @ U[..., None]).squeeze(-1)
            us.append(U)
        return torch.stack(us, 1)
