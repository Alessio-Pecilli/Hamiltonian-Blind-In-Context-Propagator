"""Windowed loader with context / anchor sampling and site-permutation augmentation (Sec. 2.7, 5.1).

Each item is a contiguous segment o_{t0 : t0+T+H} (length T+H+1) of one trajectory:
positions 0..T are the context, anchors are positions (relative to t0) where Head A is evaluated,
and the targets of anchor a are positions a+1 .. a+H of the same segment.

Segments in a batch have different T; they are right-padded. Because the body is causal, right
padding can never influence any real position, so no mask is needed.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def permute_sites(item: dict, perm: np.ndarray) -> dict:
    """Apply site permutation pi to node, edge (both axes) and metadata."""
    out = dict(item)
    out["node"] = item["node"][:, perm]
    out["edge"] = item["edge"][:, perm][:, :, perm]
    out["adj"] = item["adj"][perm][:, perm]
    out["J"] = item["J"][perm][:, perm]
    out["h"] = item["h"][perm]
    return out


class TrajectoryDataset(Dataset):
    def __init__(self, path: str, T_range: tuple[int, int], horizon: int, anchors: int = 1,
                 min_anchor: int = 8, permute: bool = True, preload: bool = True,
                 fixed_T: int | None = None, fixed_t0: int | None = None):
        """
        T_range: (Tmin, Tmax) inclusive, context length sampled uniformly per example.
        fixed_T / fixed_t0: evaluation mode (deterministic window, single anchor = T).
        """
        self.path = path
        self.T_range = T_range
        self.H = horizon
        self.anchors = anchors
        self.min_anchor = min_anchor
        self.permute = permute
        self.fixed_T = fixed_T
        self.fixed_t0 = fixed_t0
        with h5py.File(path, "r") as f:
            self.attrs = dict(f.attrs)
            self.ntraj, self.L, self.n = f["node"].shape[:3]
            self._data = {k: f[k][()] for k in f.keys()} if preload else None
        self._f = None
        Tmax = fixed_T if fixed_T is not None else T_range[1]
        if Tmax + horizon + 1 > self.L:
            raise ValueError(f"T={Tmax} + H={horizon} does not fit in L_traj={self.L}")

    def __len__(self):
        return self.ntraj

    def _get(self, key, idx, sl=slice(None)):
        if self._data is not None:
            return self._data[key][idx][sl] if key in ("node", "edge") else self._data[key][idx]
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f[key][idx, sl] if key in ("node", "edge") else self._f[key][idx]

    def __getitem__(self, idx):
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        if self.fixed_T is not None:
            T = self.fixed_T
            t0 = self.fixed_t0 if self.fixed_t0 is not None else 0
            anchors = np.array([T])
        else:
            T = int(rng.integers(self.T_range[0], self.T_range[1] + 1))
            # t0 + T + H < L_traj
            t0 = int(rng.integers(0, self.L - T - self.H))
            if T > self.min_anchor:
                extra = rng.integers(self.min_anchor, T, size=self.anchors - 1)
            else:  # spec range [t0+8, t0+T) is empty: repeat the last context position
                extra = np.full(self.anchors - 1, T)
            anchors = np.concatenate([[T], extra])
        sl = slice(t0, t0 + T + self.H + 1)
        item = dict(
            node=self._get("node", idx, sl), edge=self._get("edge", idx, sl),
            adj=self._get("adj", idx), J=self._get("J", idx), h=self._get("h", idx),
        )
        perm = rng.permutation(self.n) if self.permute else np.arange(self.n)
        item = permute_sites(item, perm)
        item.update(dt=np.float32(self._get("dt", idx)), seed=np.int64(self._get("seed", idx)),
                    T=T, t0=t0, anchors=anchors.astype(np.int64), perm=perm, index=idx)
        return item


def collate(items: list[dict]) -> dict:
    """Right-pad segments to the longest one in the batch."""
    S = max(it["node"].shape[0] for it in items)
    B, n = len(items), items[0]["node"].shape[1]
    node = np.zeros((B, S, n, 3), np.float32)
    edge = np.zeros((B, S, n, n, 9), np.float32)
    for b, it in enumerate(items):
        s = it["node"].shape[0]
        node[b, :s] = it["node"]
        edge[b, :s] = it["edge"]
    stack = lambda k, dt=None: torch.as_tensor(np.stack([it[k] for it in items]), dtype=dt)
    return dict(
        node=torch.from_numpy(node), edge=torch.from_numpy(edge),
        dt=stack("dt", torch.float32), T=stack("T", torch.long), t0=stack("t0", torch.long),
        anchors=stack("anchors", torch.long), adj=stack("adj", torch.bool),
        J=stack("J", torch.float32), h=stack("h", torch.float32), seed=stack("seed", torch.long),
        perm=stack("perm", torch.long), index=stack("index", torch.long),
    )


def gather_targets(batch: dict, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets for every anchor: node (B, A, H, n, 3), edge (B, A, H, n, n, 9)."""
    node, edge, anchors = batch["node"], batch["edge"], batch["anchors"]
    B, A = anchors.shape
    k = torch.arange(1, horizon + 1, device=anchors.device)
    idx = (anchors[:, :, None] + k).reshape(B, A * horizon)  # (B, A*H)
    bidx = torch.arange(B, device=anchors.device)[:, None]
    tn = node[bidx, idx].reshape(B, A, horizon, *node.shape[2:])
    te = edge[bidx, idx].reshape(B, A, horizon, *edge.shape[2:])
    return tn, te


def make_loader(path, T_range, horizon, anchors, batch_size, shuffle=True, num_workers=0,
                permute=True, fixed_T=None, fixed_t0=None, preload=True, min_anchor=8):
    ds = TrajectoryDataset(path, T_range, horizon, anchors, min_anchor=min_anchor, permute=permute,
                           fixed_T=fixed_T, fixed_t0=fixed_t0, preload=preload)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, drop_last=shuffle, persistent_workers=num_workers > 0)
