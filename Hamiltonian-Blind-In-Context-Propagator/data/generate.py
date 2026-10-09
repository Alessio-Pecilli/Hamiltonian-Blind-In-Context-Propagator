"""Trajectory generator (spec Sec. 2). Shot noise (Sec. 2.5) intentionally not implemented in v0.1.

Usage:
    python -m data.generate --n 8 --ntraj 1000 --L_traj 160 --dt 0.1 --family tfim \
        --seed 0 --out datasets/n8/train.h5 --workers 4
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool

import h5py
import numpy as np

from data import physics

FAMILIES = ("tfim", "xxz", "heisenberg", "lr_ising")


# ----------------------------------------------------------------------------------------
# Hamiltonian samplers
# ----------------------------------------------------------------------------------------
def is_connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    seen = np.zeros(n, bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = stack.pop()
        for j in np.nonzero(adj[i])[0]:
            if not seen[j]:
                seen[j] = True
                stack.append(j)
    return bool(seen.all())


def random_graph(n: int, rng: np.random.Generator, mean_degree: float = 3.0) -> np.ndarray:
    """Erdos-Renyi G(n, p), p = mean_degree/(n-1), resampled until connected."""
    p = min(1.0, mean_degree / (n - 1))
    while True:
        upper = np.triu(rng.random((n, n)) < p, k=1)
        adj = upper | upper.T
        if is_connected(adj):
            return adj


def sample_system(n: int, family: str, rng: np.random.Generator):
    """Returns (terms, J (n,n), h (n,), adj (n,n) bool). J/h/adj are metadata only."""
    if family == "tfim":
        adj = random_graph(n, rng)
        Jfull = rng.uniform(0.5, 1.5, size=(n, n))
        J = np.triu(Jfull, 1) * np.triu(adj, 1)
        J = J + J.T
        h = rng.uniform(0.5, 1.5, size=n)
        return physics.tfim_terms(J, h), J, h, adj

    if family in ("xxz", "heisenberg"):
        # H = sum_{(ij) in E} J_ij (X X + Y Y + Delta Z Z) + sum_i h_i Z_i
        adj = random_graph(n, rng)
        Jfull = rng.uniform(0.5, 1.5, size=(n, n))
        J = np.triu(Jfull, 1) * np.triu(adj, 1)
        J = J + J.T
        if family == "xxz":
            delta = rng.uniform(0.5, 1.5)
            h = rng.uniform(0.0, 0.5, size=n)
        else:  # disordered Heisenberg: Delta = 1, random fields
            delta = 1.0
            h = rng.uniform(-1.0, 1.0, size=n)
        terms = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    for p, c in (("X", 1.0), ("Y", 1.0), ("Z", delta)):
                        terms.append((float(J[i, j]) * c, [(i, p), (j, p)]))
        for i in range(n):
            terms.append((float(h[i]), [(i, "Z")]))
        return terms, J, h, adj

    if family == "lr_ising":
        # power-law long-range TFIM on random positions on a ring: J_ij = J0 / r_ij^alpha
        alpha = rng.uniform(1.0, 3.0)
        pos = np.sort(rng.uniform(0, n, size=n))
        d = np.abs(pos[:, None] - pos[None, :])
        d = np.minimum(d, n - d) + np.eye(n)
        J = rng.uniform(0.5, 1.5) / d**alpha
        np.fill_diagonal(J, 0.0)
        h = rng.uniform(0.5, 1.5, size=n)
        adj = ~np.eye(n, dtype=bool)
        return physics.tfim_terms(J, h), J, h, adj

    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


# ----------------------------------------------------------------------------------------
# One trajectory
# ----------------------------------------------------------------------------------------
@dataclass
class Trajectory:
    node: np.ndarray  # (L, n, 3) float32
    edge: np.ndarray  # (L, n, n, 9) float32
    dt: float
    adj: np.ndarray
    J: np.ndarray
    h: np.ndarray
    seed: int
    Psi: np.ndarray | None = None  # (L, 2^n) complex128, only if requested
    H: object = None  # sparse Hamiltonian, only if requested


def simulate(n: int, L: int, dt: float, family: str, seed: int, keep_state: bool = False,
             check: bool = True) -> Trajectory:
    """Deterministic in `seed`: regenerates the same trajectory bit-for-bit."""
    rng = np.random.default_rng(int(seed))
    terms, J, h, adj = sample_system(n, family, rng)
    psi0 = physics.random_product_state(n, rng)
    H = physics.build_hamiltonian(n, terms)
    Psi = physics.evolve(H, psi0, dt, L)
    if check:
        norm_err = np.abs(np.einsum("td,td->t", Psi.conj(), Psi).real - 1.0).max()
        E = physics.energy(H, Psi)
        e_err = np.abs(E - E[0]).max()
        if norm_err > 1e-10 or e_err > 1e-8:
            raise RuntimeError(f"seed {seed}: norm err {norm_err:.2e}, energy err {e_err:.2e}")
    node, edge = physics.observables(Psi, n)
    return Trajectory(
        node=node.astype(np.float32), edge=edge.astype(np.float32), dt=float(dt),
        adj=adj.astype(bool), J=J.astype(np.float32), h=h.astype(np.float32), seed=int(seed),
        Psi=Psi if keep_state else None, H=H if keep_state else None,
    )


def traj_seed(base_seed: int, idx: int) -> int:
    """Per-trajectory seed (stored in the file; used by the oracle baseline)."""
    return int(np.random.SeedSequence([base_seed, idx]).generate_state(1, np.uint64)[0] >> 1)


def _worker(args):
    n, L, dt, family, seed = args
    return simulate(n, L, dt, family, seed)


# ----------------------------------------------------------------------------------------
# Dataset file
# ----------------------------------------------------------------------------------------
def generate_file(out: str, n: int, ntraj: int, L: int, dt: float, family: str, seed: int,
                  workers: int = 1, verbose: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    seeds = [traj_seed(seed, k) for k in range(ntraj)]
    jobs = [(n, L, dt, family, s) for s in seeds]
    t0 = time.time()
    with h5py.File(out, "w") as f:
        f.attrs.update(dict(n=n, L_traj=L, dt=dt, family=family, base_seed=seed, ntraj=ntraj,
                            spec="propagator_spec v0.1"))
        dn = f.create_dataset("node", (ntraj, L, n, 3), "f4", chunks=(1, L, n, 3))
        de = f.create_dataset("edge", (ntraj, L, n, n, 9), "f4", chunks=(1, L, n, n, 9))
        ddt = f.create_dataset("dt", (ntraj,), "f4")
        dadj = f.create_dataset("adj", (ntraj, n, n), "bool")
        dJ = f.create_dataset("J", (ntraj, n, n), "f4")
        dh = f.create_dataset("h", (ntraj, n), "f4")
        dseed = f.create_dataset("seed", (ntraj,), "i8")
        pool = Pool(workers) if workers > 1 else None
        try:
            it = map(_worker, jobs) if pool is None else pool.imap(_worker, jobs, chunksize=4)
            for k, tr in enumerate(it):
                dn[k], de[k], ddt[k] = tr.node, tr.edge, tr.dt
                dadj[k], dJ[k], dh[k], dseed[k] = tr.adj, tr.J, tr.h, tr.seed
                if verbose and (k + 1) % max(1, ntraj // 20) == 0:
                    print(f"  {k + 1}/{ntraj}  ({time.time() - t0:.1f}s)", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    if verbose:
        print(f"wrote {out}  [{ntraj} traj, n={n}, L={L}, {time.time() - t0:.1f}s]")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--ntraj", type=int, required=True)
    p.add_argument("--L_traj", type=int, default=160)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--family", default="tfim", choices=FAMILIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    generate_file(a.out, a.n, a.ntraj, a.L_traj, a.dt, a.family, a.seed, a.workers)


if __name__ == "__main__":
    main()
