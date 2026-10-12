"""Temporal layer (Sec. 3.4): causal MHA (with RoPE) along t for each site, causal depthwise
convolution along t for each pair (plus, with pair_attn, causal MHA along t for each pair).
Never compares tokens of different sites."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embed import mlp


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding on the last dim of x (..., T, dh), positions 0..T-1."""
    T, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freqs
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half: 2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return torch.cat([out, x[..., 2 * half:]], -1) if dh % 2 else out


class TemporalLayer(nn.Module):
    def __init__(self, d: int, de: int, nh: int, kernel: int = 4, use_rope: bool = True,
                 pair_attn: bool = False):
        super().__init__()
        self.d, self.nh, self.dh, self.kernel, self.use_rope = d, nh, d // nh, kernel, use_rope
        self.ln_h = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.w_o = nn.Linear(d, d)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = mlp(d, 4 * d, d)
        self.ln_e = nn.LayerNorm(de)
        self.conv = nn.Conv1d(de, de, kernel, groups=de)  # depthwise
        self.pair_attn = pair_attn
        if pair_attn:  # causal MHA along t for every pair (same operator the sites have)
            self.pa_nh = nh if de % nh == 0 else 1
            self.pa_qkv = nn.Linear(de, 3 * de, bias=False)
            self.pa_o = nn.Linear(de, de)

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """h (B, T, n, d), e (B, T, n, n, de)."""
        B, T, n, d = h.shape
        x = self.ln_h(h).permute(0, 2, 1, 3).reshape(B * n, T, d)
        q, k, v = self.qkv(x).view(B * n, T, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        u = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B*n, nh, T, dh)
        u = u.transpose(1, 2).reshape(B, n, T, d).permute(0, 2, 1, 3)
        h = h + self.w_o(u)
        h = h + self.ff(self.ln_ff(h))
        # pairs: causal depthwise conv, kernel 4, left-padded
        de = e.shape[-1]
        y = self.ln_e(e).permute(0, 2, 3, 4, 1).reshape(B * n * n, de, T)
        y = self.conv(F.pad(y, (self.kernel - 1, 0)))
        mask = (~torch.eye(n, dtype=torch.bool, device=e.device))[None, None, :, :, None]
        e = e + y.view(B, n, n, de, T).permute(0, 4, 1, 2, 3) * mask  # keep e_ii = 0
        if self.pair_attn:
            nhp = self.pa_nh
            x = self.ln_e(e).permute(0, 2, 3, 1, 4).reshape(B * n * n, T, de)
            q, k, v = self.pa_qkv(x).view(B * n * n, T, 3, nhp, de // nhp).permute(2, 0, 3, 1, 4)
            if self.use_rope:
                q, k = rope(q), rope(k)
            u = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # (B*n*n, nhp, T, dh)
            u = u.transpose(1, 2).reshape(B, n, n, T, de).permute(0, 3, 1, 2, 4)
            e = e + self.pa_o(u) * mask
        return h, e
